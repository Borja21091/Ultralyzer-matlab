from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import logging

import cv2
import numpy as np
from PIL import Image

from backend.services.geometry_adapter import get_default_geometry_adapter
from backend.models.database import DatabaseManager, ROIDefinition


@dataclass(frozen=True)
class SegmentationBundle:
    """Segmentation masks and image metadata for one image."""

    image_id: int
    name: str
    extension: str
    seg_mask_path: Path
    shape: tuple[int, int]
    a_mask: np.ndarray
    od_mask: np.ndarray
    v_mask: np.ndarray
    vessel_mask: np.ndarray
    source_image: np.ndarray | None = None


@dataclass(frozen=True)
class LandmarkContext:
    """Image-level landmarks used by ROI generation and ROI metrics."""

    laterality: str | None = None
    disc_center_x: float | None = None
    disc_center_y: float | None = None
    disc_diameter_px: float | None = None
    disc_fovea_angle_deg: float | None = None
    fovea_center_x: float | None = None
    fovea_center_y: float | None = None


@dataclass(frozen=True)
class ROIContext:
    """Concrete ROI mask ready for metric computation."""

    roi_code: str
    roi_name: str
    generation_mode: str
    mask: np.ndarray
    area_px: float | None = None
    area_mm2: float | None = None
    mask_path: Path | None = None
    params: dict[str, Any] | None = None


class ROIMaskService:
    """Loads segmentation masks and resolves ROI definitions to boolean masks."""

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        workspace_root: Path | None = None):
        self.db_manager = db_manager or DatabaseManager()
        self.workspace_root = workspace_root or Path(__file__).resolve().parents[4]
        self.logger = logging.getLogger(self.__class__.__name__)
        self.geometry_adapter = get_default_geometry_adapter()

    def load_segmentation_bundle(self, name: str) -> SegmentationBundle:
        """Load RGB segmentation mask and split channel masks."""
        seg_metadata = self.db_manager.get_segmentation_by_filename(name)
        if not seg_metadata:
            raise ValueError(f"No segmentation data found for {name}")

        seg_mask_path = Path(seg_metadata.seg_folder) / f"{name}{seg_metadata.extension}"
        if not seg_mask_path.is_file():
            raise FileNotFoundError(f"Segmentation mask file not found: {seg_mask_path}")

        mask = np.array(Image.open(seg_mask_path))
        if mask.ndim != 3 or mask.shape[2] < 3:
            raise ValueError(f"Expected RGB segmentation mask, got shape {mask.shape}")

        a_mask = mask[:, :, 0] > 0
        od_mask = mask[:, :, 1] > 0
        v_mask = mask[:, :, 2] > 0
        vessel_mask = a_mask | v_mask

        return SegmentationBundle(
            image_id=int(seg_metadata.id),
            name=name,
            extension=str(seg_metadata.extension),
            seg_mask_path=seg_mask_path,
            shape=a_mask.shape,
            a_mask=a_mask,
            od_mask=od_mask,
            v_mask=v_mask,
            vessel_mask=vessel_mask,
        )

    def load_landmark_context(self, name: str) -> LandmarkContext:
        """Load landmark values from the database for ROI generation."""
        metrics = self.db_manager.get_landmark_metrics_by_filename(name)
        if not metrics:
            return LandmarkContext()

        return LandmarkContext(
            laterality=metrics.laterality,
            disc_center_x=metrics.disc_center_x,
            disc_center_y=metrics.disc_center_y,
            disc_diameter_px=metrics.disc_diameter_px,
            disc_fovea_angle_deg=metrics.disc_fovea_angle_deg,
            fovea_center_x=metrics.fovea_center_x,
            fovea_center_y=metrics.fovea_center_y,
        )

    def build_roi_context(
        self,
        bundle: SegmentationBundle,
        roi_definition: ROIDefinition,
        landmarks: LandmarkContext | None = None,
        compute_area_px: bool = False,
        compute_area_mm2: bool = False) -> ROIContext:
        """Build one concrete ROI mask from an ROI definition."""
        params = self._definition_params(roi_definition)
        mode = str(roi_definition.generation_mode)

        if mode == "computed_full":
            mask = self._full_mask(bundle, params)
            mask_path = None
        elif mode == "computed_central":
            if landmarks is None:
                landmarks = self.load_landmark_context(bundle.name)
            mask = self._central_mask(bundle, landmarks, params)
            mask_path = None
        elif mode == "template_mask":
            if landmarks is None:
                landmarks = self.load_landmark_context(bundle.name)
            mask, mask_path = self._template_mask(bundle, landmarks, params)
        else:
            raise ValueError(f"Unsupported ROI generation mode: {mode}")

        area_px = None
        if compute_area_px or compute_area_mm2:
            area_px = float(np.sum(mask))
        area_mm2 = None
        if compute_area_mm2:
            area_mm2 = self._roi_area_mm2(mask, bundle.name, str(roi_definition.code))

        return ROIContext(
            roi_code=str(roi_definition.code),
            roi_name=str(roi_definition.name),
            generation_mode=mode,
            mask=mask,
            area_px=area_px,
            area_mm2=area_mm2,
            mask_path=mask_path,
            params=params,
        )

    def _roi_area_mm2(self, mask: np.ndarray, name: str, roi_code: str) -> float | None:
        if not np.any(mask):
            return None
        try:
            return float(self.geometry_adapter.roi_area(
                np.argwhere(mask).astype(float),
                area_algorithm="cross_product",
            ))
        except Exception as e:
            self.logger.error(f"Error converting ROI '{roi_code}' area to mm2 for {name}: {str(e)}")
            return None

    def _definition_params(self, roi_definition: ROIDefinition) -> dict[str, Any]:
        if not roi_definition.params_json:
            return {}
        return json.loads(str(roi_definition.params_json))

    def _full_mask(self, bundle: SegmentationBundle, params: dict[str, Any]) -> np.ndarray:
        image = bundle.source_image if bundle.source_image is not None else self._load_source_image(bundle.name)
        if image.shape[:2] != bundle.shape:
            image = cv2.resize(
                image,
                (bundle.shape[1], bundle.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        lightness_mask = (hls_image[:, :, 1] > 0).astype(np.uint8)

        kernel_size = int(params.get("close_kernel_px", 11))
        kernel_size = max(1, kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed_mask = cv2.morphologyEx(lightness_mask, cv2.MORPH_CLOSE, kernel)

        return self._largest_connected_component(closed_mask)

    def _central_mask(
        self,
        bundle: SegmentationBundle,
        landmarks: LandmarkContext,
        params: dict[str, Any]) -> np.ndarray:
        if landmarks.disc_center_x is None or landmarks.disc_center_y is None:
            raise ValueError("Central ROI requires optic disc center")

        radius_px = float(params.get("radius_px", 400.0))
        rows, cols = np.ogrid[:bundle.shape[0], :bundle.shape[1]]
        distance_sq = (
            (cols - float(landmarks.disc_center_x)) ** 2
            + (rows - float(landmarks.disc_center_y)) ** 2
        )
        return distance_sq <= radius_px ** 2

    def _template_mask(
        self,
        bundle: SegmentationBundle,
        landmarks: LandmarkContext,
        params: dict[str, Any]) -> tuple[np.ndarray, Path]:
        if landmarks.laterality not in {"left", "right"}:
            raise ValueError("Template ROI requires left/right laterality")

        template_key = f"{landmarks.laterality}_template"
        template_path_value = params.get(template_key)
        if not template_path_value:
            raise ValueError(f"Template ROI params missing '{template_key}'")

        template_path = self._resolve_path(str(template_path_value))
        if not template_path.is_file():
            raise FileNotFoundError(f"ROI template mask not found: {template_path}")

        template = Image.open(template_path).convert("L")
        expected_size = (bundle.shape[1], bundle.shape[0])
        if template.size != expected_size:
            template = template.resize(expected_size, Image.Resampling.NEAREST)

        return np.array(template) > 0, template_path

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.workspace_root / path

    def _load_source_image(self, name: str) -> np.ndarray:
        metadata = self.db_manager.get_metadata_by_filename(name)
        if not metadata:
            raise ValueError(f"No source image metadata found for {name}")

        image_path = Path(str(metadata.folder)) / f"{metadata.name}{metadata.extension}"
        if not image_path.is_file():
            raise FileNotFoundError(f"Source image not found: {image_path}")

        return np.array(Image.open(image_path).convert("RGB"))

    def _largest_connected_component(self, mask: np.ndarray) -> np.ndarray:
        binary_mask = mask.astype(np.uint8)
        if not np.any(binary_mask):
            return np.zeros_like(binary_mask, dtype=bool)

        num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
        if num_labels <= 1:
            return binary_mask.astype(bool)

        label_sizes = np.bincount(labels.ravel())
        label_sizes[0] = 0
        largest_label = int(np.argmax(label_sizes))
        return labels == largest_label