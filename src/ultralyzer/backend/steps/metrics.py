from pathlib import Path
from typing import Any, Callable
import json
import logging

from backend.models.database import DatabaseManager
from backend.services.geometry_adapter import GeometryReadiness, get_default_geometry_adapter
from backend.services.metric_calculators import LandmarkMetricsCalculator, ROIMetricsCalculator
from backend.services.roi_masks import LandmarkContext, ROIMaskService, SegmentationBundle
from backend.steps.base_step import ProcessingStep


class MetricsStep(ProcessingStep):
    """Orchestrates landmark and ROI-dependent metric calculation."""

    def __init__(
        self,
        db_manager: DatabaseManager = None):

        super().__init__("Metrics Calculation", 3)
        self.db_manager = db_manager or DatabaseManager()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.geometry_adapter = get_default_geometry_adapter()
        self.roi_mask_service = ROIMaskService(self.db_manager)
        self.landmark_calculator = LandmarkMetricsCalculator(self.logger)
        self.roi_calculator = ROIMetricsCalculator(self.logger)

        self.a_mask = None
        self.v_mask = None
        self.od_mask = None
        self.vessel_mask = None

    def geometry_readiness(self) -> GeometryReadiness:
        return self.geometry_adapter.readiness()

    def process(
        self,
        image_path: str,
        extension: str = ".png",
        skip_existing_roi_metrics: bool = False,
        status_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Calculate landmark metrics once and ROI metrics for selected ROIs."""
        _ = extension
        if not self.validate_input(Path(image_path)):
            self._emit_status(status_callback, f"Metrics error for {Path(image_path).name}: Invalid input")
            return {"success": False, "error": "Invalid input"}

        try:
            name = Path(image_path).stem
            bundle = self.roi_mask_service.load_segmentation_bundle(name)
            self._set_current_masks(bundle)
            geometry_readiness = self.geometry_readiness()

            self._emit_status(status_callback, f"Metrics: {Path(image_path).name} -> landmark metrics")

            landmark_metrics = self.landmark_calculator.compute(
                bundle,
                self.db_manager.get_fovea_by_filename(name)
            )
            if not self.db_manager.save_landmark_metrics_by_id(bundle.image_id, landmark_metrics):
                self._emit_status(status_callback, f"Metrics error for {Path(image_path).name}: Failed to save landmark metrics")
                return {"success": False, "error": "Failed to save landmark metrics"}

            landmark_context = self.landmark_calculator.to_context(landmark_metrics)
            roi_result = self._process_selected_rois(
                bundle,
                landmark_context,
                skip_existing=skip_existing_roi_metrics,
                status_callback=status_callback,
            )
            success = not roi_result["errors"] and (
                bool(roi_result["metrics"]) or bool(roi_result["skipped"])
            )

            return {
                "success": success,
                "geometry_readiness": geometry_readiness,
                "landmark_metrics": landmark_metrics,
                "roi_metrics": roi_result["metrics"],
                "roi_skipped": roi_result["skipped"],
                "roi_errors": roi_result["errors"],
                "metrics": {
                    **landmark_metrics,
                    "roi_metrics": roi_result["metrics"],
                    "roi_skipped": roi_result["skipped"],
                },
            }

        except Exception as e:
            self.logger.error(f"Error calculating metrics for {Path(image_path).name}: {str(e)}")
            self._emit_status(status_callback, f"Metrics error for {Path(image_path).name}: {str(e)}")
            return {"success": False, "error": str(e)}

    def process_and_save_to_db(
        self,
        image_path: str,
        id: int,
        skip_existing_roi_metrics: bool = False,
        status_callback: Callable[[str], None] | None = None) -> bool:
        """Process image and save landmark/ROI metric results to database."""
        seg_meta = self.db_manager.get_segmentation_result_by_id(id)
        if not seg_meta:
            self.logger.error(f"No segmentation data found for ID {id}. Cannot calculate metrics.")
            self._emit_status(status_callback, f"Metrics error for {Path(image_path).name}: No segmentation data found")
            return False

        result = self.process(
            image_path,
            seg_meta.extension,
            skip_existing_roi_metrics=skip_existing_roi_metrics,
            status_callback=status_callback,
        )
        if not result["success"]:
            detail = result.get("error")
            if not detail and result.get("roi_errors"):
                detail = "; ".join(
                    f"{roi_code}: {error}"
                    for roi_code, error in result["roi_errors"].items()
                )
            self.logger.error(f"Processing failed for {image_path}: {detail}")
            if detail:
                self._emit_status(status_callback, f"Metrics failed for {Path(image_path).name}: {detail}")
            return False

        return True

    def get_pending_images(self):
        """Get all images that need metrics calculation."""
        metadata = self.db_manager.get_pending_metrics()
        return sorted(metadata, key=lambda x: x.name)

    def _process_selected_rois(
        self,
        bundle: SegmentationBundle,
        landmarks: LandmarkContext,
        skip_existing: bool = False,
        status_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
        metrics_by_roi: dict[str, Any] = {}
        skipped_by_roi: dict[str, str] = {}
        errors_by_roi: dict[str, str] = {}
        image_rois = self._selected_image_rois(bundle.image_id)

        for image_roi in image_rois:
            roi_definition = image_roi.roi_definition
            roi_code = str(roi_definition.code)
            try:
                self._emit_status(status_callback, f"Metrics: {bundle.name} -> ROI {roi_code}")
                image_roi_id = int(str(image_roi.id))
                if skip_existing and self.db_manager.has_roi_metrics(image_roi_id):
                    skipped_by_roi[roi_code] = "Already computed"
                    self._emit_status(status_callback, f"Metrics: {bundle.name} -> ROI {roi_code} skipped (Already computed)")
                    continue

                roi_context = self.roi_mask_service.build_roi_context(
                    bundle=bundle,
                    roi_definition=roi_definition,
                    landmarks=landmarks,
                    compute_area_px=True,
                    compute_area_mm2=True,
                )
                image_roi_id = self.db_manager.ensure_image_roi(
                    image_id=bundle.image_id,
                    roi_code=roi_code,
                    selected_for_metrics=True,
                    mask_path=str(roi_context.mask_path) if roi_context.mask_path else None,
                    geometry_json=json.dumps({
                        "generation_mode": roi_context.generation_mode,
                        "params": roi_context.params or {},
                    }, sort_keys=True),
                    area_px=roi_context.area_px,
                    area_mm2=roi_context.area_mm2,
                )
                if image_roi_id is None:
                    errors_by_roi[roi_code] = "Failed to persist image ROI"
                    self._emit_status(status_callback, f"Metrics error for {bundle.name} ROI {roi_code}: Failed to persist image ROI")
                    continue

                if skip_existing and self.db_manager.has_roi_metrics(image_roi_id):
                    skipped_by_roi[roi_code] = "Already computed"
                    self._emit_status(status_callback, f"Metrics: {bundle.name} -> ROI {roi_code} skipped (Already computed)")
                    continue

                roi_metrics = self.roi_calculator.compute(bundle, roi_context, landmarks)
                if not self.db_manager.save_roi_metrics(image_roi_id, roi_metrics):
                    errors_by_roi[roi_code] = "Failed to save ROI metrics"
                    self._emit_status(status_callback, f"Metrics error for {bundle.name} ROI {roi_code}: Failed to save ROI metrics")
                    continue

                metrics_by_roi[roi_code] = roi_metrics
                self._emit_status(status_callback, f"Metrics: {bundle.name} -> ROI {roi_code} complete")
            except Exception as e:
                self.logger.error(f"Error calculating ROI '{roi_code}' for {bundle.name}: {str(e)}")
                errors_by_roi[roi_code] = str(e)
                self._emit_status(status_callback, f"Metrics error for {bundle.name} ROI {roi_code}: {str(e)}")

        return {"metrics": metrics_by_roi, "skipped": skipped_by_roi, "errors": errors_by_roi}

    def _emit_status(self, status_callback: Callable[[str], None] | None, message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    def _selected_image_rois(self, image_id: int):
        image_rois = self.db_manager.get_image_rois_by_image_id(
            image_id=image_id,
            selected_only=True,
        )
        if image_rois:
            return image_rois

        self.db_manager.ensure_image_roi(
            image_id=image_id,
            roi_code="full",
            selected_for_metrics=True,
        )
        return self.db_manager.get_image_rois_by_image_id(
            image_id=image_id,
            selected_only=True,
        )

    def _set_current_masks(self, bundle: SegmentationBundle) -> None:
        self.a_mask = bundle.a_mask
        self.v_mask = bundle.v_mask
        self.od_mask = bundle.od_mask
        self.vessel_mask = bundle.vessel_mask