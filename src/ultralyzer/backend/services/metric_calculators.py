from typing import Any
import logging

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from backend.services.geometry_adapter import get_default_geometry_adapter
from backend.services.roi_masks import LandmarkContext, ROIContext, SegmentationBundle
from backend.utils.arcades import ArcadeRANSAC
from backend.utils.feature_measurement import calculate_vessel_widths_mm, calculate_vessel_widths_px
from backend.utils.feature_measurement import chord_length, compute_edges, curve_length
from backend.utils.feature_measurement import fractal_dimension_boxcount, fractal_dimension_sandbox
from backend.utils.feature_measurement import generate_vessel_skeleton, pair_distances, tortuosity_density
from backend.utils.preprocessing import localise_centre_mass


class LandmarkMetricsCalculator:
    """Computes image-level disc/fovea metrics independent of ROI selection."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.geometry_adapter = get_default_geometry_adapter()

    def compute(
        self,
        bundle: SegmentationBundle,
        fovea_center: tuple[float | None, float | None]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        disc_flag = bool(bundle.od_mask.any())

        if not disc_flag:
            self.logger.warning(f"No optic disc detected in {bundle.name}. Skipping OD metrics.")
        else:
            od_center_y, od_center_x = localise_centre_mass((bundle.od_mask * 255).astype(np.uint8))
            metrics["disc_center_x"] = float(od_center_x)
            metrics["disc_center_y"] = float(od_center_y)
            metrics["disc_area_px"] = float(np.sum(bundle.od_mask))
            metrics["disc_diameter_px"] = float(2 * np.sqrt(metrics["disc_area_px"] / np.pi))

            labeled_od = label(bundle.od_mask)
            props = regionprops(labeled_od)[0]
            metrics["disc_major_axis_px"] = float(props.axis_major_length)
            metrics["disc_minor_axis_px"] = float(props.axis_minor_length)
            metrics["disc_orientation_deg"] = float(props.orientation * (180.0 / np.pi))
            metrics["disc_eccentricity"] = float(props.eccentricity)
            metrics["disc_circularity"] = (
                float((4 * np.pi * props.area) / (props.perimeter ** 2))
                * float((1 - 0.5 / ((props.perimeter / (2 * np.pi)) + 0.5)) ** 2)
                if props.perimeter > 0 else 0.0
            )

            metrics.update(self._compute_disc_um_metrics(metrics))
            try:
                metrics["disc_area_mm2"] = self.geometry_adapter.roi_area(
                    np.argwhere(bundle.od_mask).astype(float),
                    area_algorithm="cross_product"
                )
            except Exception as e:
                self.logger.error(f"Error converting optic disc area to mm2: {str(e)}")

        fovea_center_x, fovea_center_y = fovea_center
        fovea_flag = fovea_center_x is not None and fovea_center_y is not None
        if fovea_flag:
            metrics["fovea_center_x"] = float(fovea_center_x)
            metrics["fovea_center_y"] = float(fovea_center_y)
        else:
            self.logger.warning(
                f"Fovea location not found in database for {bundle.name}. "
                "Identify the fovea with the edit-mask tool and re-run metric calculation."
            )

        if fovea_flag and disc_flag:
            fov_x = float(fovea_center_x)
            fov_y = float(fovea_center_y)
            disc_x = float(metrics["disc_center_x"])
            disc_y = float(metrics["disc_center_y"])
            od_fovea_distance = np.sqrt((disc_x - fov_x) ** 2 + (disc_y - fov_y) ** 2)
            metrics["disc_fovea_distance_px"] = float(od_fovea_distance)

            fov_y_cart = bundle.shape[0] - fov_y
            disc_y_cart = bundle.shape[0] - disc_y
            metrics["disc_fovea_angle_deg"] = float(np.degrees(np.arctan(
                (fov_y_cart - disc_y_cart) / np.abs(fov_x - disc_x + 1e-6)
            )))
            metrics["laterality"] = "right" if fov_x < disc_x else "left"

            try:
                metrics["disc_fovea_distance_um"] = float(
                    np.ravel(pair_distances(
                        np.array([[disc_y, disc_x]]).astype(np.float64),
                        np.array([[fov_y, fov_x]]).astype(np.float64)
                    ) * 1e3)[0]
                )
            except Exception as e:
                self.logger.error(f"Error converting disc-fovea distance to microns: {str(e)}")

        return metrics

    def to_context(self, metrics: dict[str, Any]) -> LandmarkContext:
        return LandmarkContext(
            laterality=metrics.get("laterality"),
            disc_center_x=metrics.get("disc_center_x"),
            disc_center_y=metrics.get("disc_center_y"),
            disc_diameter_px=metrics.get("disc_diameter_px"),
            disc_fovea_angle_deg=metrics.get("disc_fovea_angle_deg"),
            fovea_center_x=metrics.get("fovea_center_x"),
            fovea_center_y=metrics.get("fovea_center_y"),
        )

    def _compute_disc_um_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        disc_um_metrics = {}
        try:
            radius_px = metrics["disc_diameter_px"] / 2.0
            edge_points_px = np.array([
                np.array([metrics["disc_center_y"], metrics["disc_center_x"]])
                + radius_px * np.array([np.cos(theta), -np.sin(theta)])
                for theta in np.linspace(0, 2 * np.pi, num=360, endpoint=False)
            ])
            count = edge_points_px.shape[0]

            disc_diameter_um = pair_distances(
                edge_points_px[0:count // 2 - 1, :],
                edge_points_px[count // 2:count - 1, :]
            ) * 1e3
            disc_diameter_um = np.mean(disc_diameter_um)

            orientation_rad = np.radians(metrics["disc_orientation_deg"])
            major_axis_coords_px = np.array([
                [metrics["disc_center_x"] - (metrics["disc_major_axis_px"] / 2) * np.sin(orientation_rad),
                 metrics["disc_center_y"] + (metrics["disc_major_axis_px"] / 2) * np.cos(orientation_rad)],
                [metrics["disc_center_x"] + (metrics["disc_major_axis_px"] / 2) * np.sin(orientation_rad),
                 metrics["disc_center_y"] - (metrics["disc_major_axis_px"] / 2) * np.cos(orientation_rad)]
            ])
            minor_axis_coords_px = np.array([
                [metrics["disc_center_x"] - (metrics["disc_minor_axis_px"] / 2) * np.cos(orientation_rad),
                 metrics["disc_center_y"] - (metrics["disc_minor_axis_px"] / 2) * np.sin(orientation_rad)],
                [metrics["disc_center_x"] + (metrics["disc_minor_axis_px"] / 2) * np.cos(orientation_rad),
                 metrics["disc_center_y"] + (metrics["disc_minor_axis_px"] / 2) * np.sin(orientation_rad)]
            ])
            major_axis_um = pair_distances(
                major_axis_coords_px[0, :].astype(np.float64).reshape(1, -1),
                major_axis_coords_px[1, :].astype(np.float64).reshape(1, -1)
            ) * 1e3
            minor_axis_um = pair_distances(
                minor_axis_coords_px[0, :].astype(np.float64).reshape(1, -1),
                minor_axis_coords_px[1, :].astype(np.float64).reshape(1, -1)
            ) * 1e3
            
            disc_um_metrics = {
                "disc_diameter_um": float(disc_diameter_um),
                "disc_major_axis_um": float(np.ravel(major_axis_um)[0]),
                "disc_minor_axis_um": float(np.ravel(minor_axis_um)[0]),
            }
        except Exception as e:
            self.logger.error(f"Error converting optic disc metrics to microns: {str(e)}")

        return disc_um_metrics


class AVMetricsCalculator:
    """Computes artery & vein metrics that are independent of ROI selection."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def compute(
        self, 
        bundle: SegmentationBundle, 
        landmarks: LandmarkContext) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        
        disc_center_x = landmarks.disc_center_x
        disc_center_y = landmarks.disc_center_y
        disc_diameter_px = landmarks.disc_diameter_px

        artery_mask = bundle.a_mask.astype(bool)
        vein_mask = bundle.v_mask.astype(bool)
        
        if not artery_mask.any() or not vein_mask.any():
            self.logger.warning(f"No vessels (arteries or veins) detected in {bundle.name}. Skipping vessel metrics.")
            return metrics

        if disc_center_x is None or disc_center_y is None or disc_diameter_px is None:
            self.logger.warning(f"Optic disc metrics missing for {bundle.name}. Skipping CRAE/CRVE metrics.")
            return metrics
        
        # CRAE and CRVE ring mask
        yy, xx = np.ogrid[:bundle.shape[0], :bundle.shape[1]]
        dist_sq = (yy - disc_center_y)**2 + (xx - disc_center_x)**2
        ring_mask = (dist_sq >= disc_diameter_px**2) & (dist_sq <= (1.5 * disc_diameter_px)**2)
        
        artery_mask *= ring_mask
        vein_mask *= ring_mask
        zonal_vessels = [artery_mask, vein_mask]
        
        # CRAE and CRVE
        for vtype, mask in zip(["artery", "vein"], zonal_vessels):
            if not mask.any():
                self.logger.warning(f"No {vtype}s detected in CRAE/CRVE ring for {bundle.name}. Skipping {vtype} caliber metrics.")
                continue
            
            vcoords = generate_vessel_skeleton(mask.astype(np.uint8), bundle.od_mask, (disc_center_y, disc_center_x))
            edges1, edges2, _ = compute_edges(mask.astype(np.uint8), vcoords)
            _, avg_vessel_widths_px = calculate_vessel_widths_px(edges1, edges2)
            _, avg_vessel_widths_mm = calculate_vessel_widths_mm(edges1, edges2)
            
            N_vessels = len(avg_vessel_widths_mm)
            
            if N_vessels < 6:
                self.logger.warning(f"Only {N_vessels} {vtype}s detected in CRAE/CRVE ring for {bundle.name}. Expected 6 for Knudtson caliber calculation.")
                continue
            else:
                caliber_px = self._caliber_algorithm(avg_vessel_widths_px, vtype)
                caliber_mm = self._caliber_algorithm(avg_vessel_widths_mm, vtype)
            
            metrics[f"cr{vtype[0]}e_px"] = float(caliber_px)
            metrics[f"cr{vtype[0]}e_um"] = float(caliber_mm) * 1e3  # Convert mm to microns
        
        # AVR
        if "crae_px" in metrics and "crve_px" in metrics and metrics["crve_px"] != 0:
            metrics["avr_px"] = float(metrics["crae_px"] / metrics["crve_px"])
        if "crae_um" in metrics and "crve_um" in metrics and metrics["crve_um"] != 0:
            metrics["avr_um"] = float(metrics["crae_um"] / metrics["crve_um"])
        
        return metrics
    
    def _caliber_algorithm(
        self, 
        widths: list[float], 
        vtype: str
        ) -> float:
        
        while len(widths) > 1:
            # Sort from biggest to lowest and pick the first 6
            widths = sorted(widths, reverse=True)[:6]
            
            caliber = []
            for i in range(len(widths) // 2):
                w1 = widths[i]
                w2 = widths[-(i + 1)]
                caliber.append(self._knudtson_caliber(w1, w2, vtype))
            if len(widths) % 2 == 1:
                caliber.append(widths[len(widths) // 2])  # Add the middle one if odd
            widths = caliber
            
        return float(widths[0]) if widths else float("nan")
    
    def _knudtson_caliber(self, w1: float, w2: float, vtype: str) -> float:
        if vtype.lower() not in ["artery", "vein"]:
            self.logger.warning(f"Unknown vessel type '{vtype}' for Knudtson caliber calculation. Returning NaN.")
            return float("nan")
        
        k = 0.88 if vtype.lower() == "artery" else 0.95
        return k * (w1**2 + w2**2)**0.5


class ROIMetricsCalculator:
    """Computes vessel metrics for one concrete ROI mask."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.geometry_adapter = get_default_geometry_adapter()

    def compute(
        self,
        bundle: SegmentationBundle,
        roi: ROIContext,
        landmarks: LandmarkContext) -> dict[str, Any]:
        roi_mask = roi.mask.astype(bool)
        roi_area_px = float(np.sum(roi_mask))
        if roi_area_px <= 0:
            raise ValueError(f"ROI '{roi.roi_code}' is empty for {bundle.name}")

        metrics: dict[str, Any] = {}
        masks = [
            bundle.vessel_mask & roi_mask,
            bundle.a_mask & roi_mask,
            bundle.v_mask & roi_mask,
        ]

        od_center = self._od_center(landmarks)
        for vessel_mask, prefix in zip(masks, ["vessel", "a", "v"]):
            if not vessel_mask.any():
                label_name = "vessels" if prefix == "vessel" else ("arteries" if prefix == "a" else "veins")
                self.logger.warning(f"No {label_name} detected in {bundle.name} ROI '{roi.roi_code}'.")
                continue

            metrics[f"{prefix}_density"] = float(np.sum(vessel_mask) / roi_area_px)
            metrics[f"{prefix}_fractal_dimension_sandbox"] = float(
                fractal_dimension_sandbox(vessel_mask.astype(int))
            )
            metrics[f"{prefix}_fractal_dimension_boxcount"] = float(
                fractal_dimension_boxcount(vessel_mask.astype(int))
            )
            try:
                metrics[f"{prefix}_area_mm2"] = self.geometry_adapter.roi_area(
                    np.argwhere(vessel_mask).astype(float),
                    area_algorithm="cross_product"
                )
                if roi.area_mm2 is not None and roi.area_mm2 > 0:
                    metrics[f"{prefix}_density_area"] = float(
                        metrics[f"{prefix}_area_mm2"] / roi.area_mm2
                    )
            except Exception as e:
                self.logger.error(f"Error converting {prefix} area to mm2 in ROI '{roi.roi_code}': {str(e)}")

            if od_center is not None:
                metrics.update(self._compute_skeleton_metrics(bundle, vessel_mask, prefix, od_center, roi.roi_code))

        if "a_density" in metrics and "v_density" in metrics:
            metrics["av_ratio_px"] = metrics["a_density"] / metrics["v_density"] if metrics["v_density"] > 0 else float("nan")

        if "a_area_mm2" in metrics and "v_area_mm2" in metrics:
            metrics["av_ratio_um"] = metrics["a_area_mm2"] / metrics["v_area_mm2"] if metrics["v_area_mm2"] > 0 else float("nan")
        
        if masks[1].any() and masks[2].any():
            artery_skeleton = skeletonize(masks[1])
            vein_skeleton = skeletonize(masks[2])
            metrics["av_crossings"] = float(np.sum(artery_skeleton & vein_skeleton))

        metrics["av_arcade_concavity"] = None
        if roi.roi_code == "central" and masks[1].any() and masks[2].any():
            arcade_concavity = self._compute_arcade_concavity(bundle, masks[0], landmarks, roi.roi_code)
            if arcade_concavity is not None:
                metrics["av_arcade_concavity"] = arcade_concavity

        return metrics

    def _od_center(self, landmarks: LandmarkContext) -> tuple[float, float] | None:
        if landmarks.disc_center_y is None or landmarks.disc_center_x is None:
            return None
        return (float(landmarks.disc_center_y), float(landmarks.disc_center_x))

    def _compute_skeleton_metrics(
        self,
        bundle: SegmentationBundle,
        vessel_mask: np.ndarray,
        prefix: str,
        od_center: tuple[float, float],
        roi_code: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        try:
            vcoords = generate_vessel_skeleton(vessel_mask.astype(np.uint8), bundle.od_mask, od_center)
            if not vcoords:
                return metrics

            tcc = 0.0
            td = 0.0
            zonal_vessels = []
            for vessel in vcoords:
                zonal_vessels.append(vessel)
                vessel_t = vessel.T
                vessel_length = curve_length(vessel_t[1], vessel_t[0])
                vessel_chord = chord_length(vessel_t[1], vessel_t[0])
                if vessel_chord > 0:
                    tcc += vessel_length / vessel_chord
                td += tortuosity_density(vessel_t[1], vessel_t[0], vessel_length)

            vessel_count = len(vcoords)
            metrics[f"{prefix}_tortuosity_density"] = float(td / vessel_count)
            metrics[f"{prefix}_tortuosity_distance"] = float(tcc / vessel_count)

            edges1, edges2, centerline_coords = compute_edges(vessel_mask.astype(np.uint8), zonal_vessels)
            all_vessel_widths, avg_width = calculate_vessel_widths_px(edges1, edges2)
            all_vessel_widths_mm, avg_width_mm = calculate_vessel_widths_mm(edges1, edges2)

            if len(avg_width):
                metrics[f"{prefix}_width_px"] = float(np.mean(avg_width))
            if len(avg_width_mm):
                metrics[f"{prefix}_width_um"] = float(np.mean(avg_width_mm) * 1e3)

            if centerline_coords.size:
                dist = np.linalg.norm(centerline_coords - np.array([[od_center[0], od_center[1]]]), axis=1)
                dist_mm = pair_distances(
                    centerline_coords[:, ::-1].astype(np.float64),
                    np.tile(np.array([[od_center[1], od_center[0]]]), (centerline_coords.shape[0], 1)).astype(np.float64),
                )

                width_values = np.concatenate(all_vessel_widths) if all_vessel_widths else np.array([])
                width_values_mm = np.concatenate(all_vessel_widths_mm) if all_vessel_widths_mm else np.array([])
                if len(dist) >= 2 and len(width_values) == len(dist):
                    p = np.polyfit(dist, width_values, 1)
                    metrics[f"{prefix}_width_gradient_px"] = float(p[0])
                    metrics[f"{prefix}_width_intercept_px"] = float(p[1])
                if len(dist_mm) >= 2 and len(width_values_mm) == len(dist_mm):
                    p_mm = np.polyfit(dist_mm, width_values_mm, 1)
                    metrics[f"{prefix}_width_gradient_um"] = float(p_mm[0]) * 1e3
                    metrics[f"{prefix}_width_intercept_um"] = float(p_mm[1]) * 1e3
        except Exception as e:
            self.logger.error(
                f"Error calculating {prefix} skeleton metrics for {bundle.name} ROI '{roi_code}': {str(e)}"
            )

        return metrics

    def _compute_arcade_concavity(
        self,
        bundle: SegmentationBundle,
        vessel_mask: np.ndarray,
        landmarks: LandmarkContext,
        roi_code: str) -> float | None:
        required = [
            landmarks.disc_center_x,
            landmarks.disc_center_y,
            landmarks.disc_diameter_px,
            landmarks.disc_fovea_angle_deg,
            landmarks.laterality,
        ]
        if any(value is None for value in required):
            return None

        try:
            arcader = ArcadeRANSAC(
                name=bundle.name,
                mask=vessel_mask,
                disc_center_x=float(landmarks.disc_center_x),
                disc_center_y=float(landmarks.disc_center_y),
                disc_diameter_px=float(landmarks.disc_diameter_px),
                disc_fovea_angle_deg=float(landmarks.disc_fovea_angle_deg),
                laterality=str(landmarks.laterality),
            )
            arcader()
            return float(np.abs(arcader.concavity)) if arcader.concavity is not None else None
        except Exception as e:
            self.logger.error(f"Error calculating arcade concavity for {bundle.name} ROI '{roi_code}': {str(e)}")
            return None