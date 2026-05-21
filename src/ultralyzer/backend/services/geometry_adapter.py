from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
import logging
import os
from typing import Any, Iterable, Sequence

import numpy as np


MATLAB_SPHERE_RADIUS_UM = 12000.0


class GeometryCapability(str, Enum):
    DISTANCE = "distance"
    AREA = "area"


class GeometryTrafficLight(str, Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


class GeometryCapabilityUnavailableError(RuntimeError):
    """Raised when no configured backend can serve a requested capability."""


@dataclass(frozen=True)
class GeometryCapabilities:
    distance: bool = False
    area: bool = False

    def supports(self, capability: GeometryCapability) -> bool:
        if capability is GeometryCapability.DISTANCE:
            return self.distance
        if capability is GeometryCapability.AREA:
            return self.area
        return False

    def available(self) -> tuple[GeometryCapability, ...]:
        return tuple(
            capability
            for capability in GeometryCapability
            if self.supports(capability)
        )

    def missing(self) -> tuple[GeometryCapability, ...]:
        return tuple(
            capability
            for capability in GeometryCapability
            if not self.supports(capability)
        )

    def is_empty(self) -> bool:
        return not any((self.distance, self.area))

    def is_complete(self) -> bool:
        return all((self.distance, self.area))


@dataclass(frozen=True)
class GeometryBackendStatus:
    name: str
    capabilities: GeometryCapabilities
    detail: str | None = None


@dataclass(frozen=True)
class GeometryReadiness:
    traffic_light: GeometryTrafficLight
    capabilities: GeometryCapabilities
    distance_backend: str | None
    area_backend: str | None
    backend_statuses: tuple[GeometryBackendStatus, ...]

    def missing_metrics(self) -> tuple[str, ...]:
        return missing_geometry_metrics(self.capabilities)

    def summary(self) -> str:
        if self.traffic_light is GeometryTrafficLight.GREEN:
            return "Geometry backend ready for all supported computations."
        if self.traffic_light is GeometryTrafficLight.YELLOW:
            return (
                "Geometry backend partially ready; some geometry-dependent metrics "
                "will be skipped."
            )
        return "No geometry backend is available; geometry-dependent metrics will be skipped."


class GeometryBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def probe(self) -> GeometryBackendStatus:
        """Return the backend readiness and supported capabilities."""

    def pair_distances(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        raise GeometryCapabilityUnavailableError(
            f"Backend '{self.name}' does not support {GeometryCapability.DISTANCE.value}."
        )

    def roi_area(
        self,
        points: np.ndarray,
        area_algorithm: str = "cross_product",
    ) -> float:
        raise GeometryCapabilityUnavailableError(
            f"Backend '{self.name}' does not support {GeometryCapability.AREA.value}."
        )


class MatlabGeometryBackend(GeometryBackend):
    @property
    def name(self) -> str:
        return "matlab"

    def probe(self) -> GeometryBackendStatus:
        engine, bin_dir = self._matlab_context()
        distance_binary = os.path.join(bin_dir, "distanceOnSP.p")
        distance_ready = engine is not None and os.path.exists(distance_binary)

        details: list[str] = []
        if not os.path.exists(distance_binary):
            details.append("distanceOnSP.p not found")
        if engine is None:
            details.append("MATLAB engine unavailable")

        return GeometryBackendStatus(
            name=self.name,
            capabilities=GeometryCapabilities(distance=distance_ready, area=distance_ready),
            detail="; ".join(details) if details else None,
        )

    def pair_distances(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        engine, bin_dir = self._matlab_context()
        distance_binary = os.path.join(bin_dir, "distanceOnSP.p")
        if engine is None or not os.path.exists(distance_binary):
            raise GeometryCapabilityUnavailableError(
                "MATLAB geometry backend is not ready for geodesic distance support."
            )
        if len(p1) == 0:
            return np.empty(0, dtype=np.float64)

        p1 = np.ascontiguousarray(p1 + 1, dtype=np.float64)[:, ::-1]
        p2 = np.ascontiguousarray(p2 + 1, dtype=np.float64)[:, ::-1]
        engine.cd(str(bin_dir))
        if os.path.exists(os.path.join(bin_dir, "batchDistanceOnSP.m")):
            return self._batch_pair_distances(engine, p1, p2)
        distances = [engine.distanceOnSP(xs, ys, xf, yf) for (xs, ys), (xf, yf) in zip(p1, p2)]
        return np.asarray(distances, dtype=np.float64)

    def roi_area(
        self,
        points: np.ndarray,
        area_algorithm: str = "cross_product",
    ) -> float:
        if points.size == 0:
            return 0.0

        points = np.ascontiguousarray(points, dtype=np.float64)
        p1 = points
        p2 = points + np.array([0, 1])
        p3 = points + np.array([1, 0])
        p4 = points + np.array([1, 1])

        segment_count = len(points)
        segment_starts = np.concatenate((p1, p1, p2, p2, p3), axis=0)
        segment_ends = np.concatenate((p2, p3, p3, p4, p4), axis=0)
        segment_distances = self._deduplicated_pair_distances(segment_starts, segment_ends)
        d12, d13, d23, d24, d34 = np.split(
            segment_distances,
            (segment_count, 2 * segment_count, 3 * segment_count, 4 * segment_count),
        )

        area1 = self._spherical_triangle_area_from_side_lengths(d12, d13, d23)
        area2 = self._spherical_triangle_area_from_side_lengths(d24, d34, d23)
        total_area = np.sum(area1 + area2, dtype=np.float64)
        return float(total_area)

    def _batch_pair_distances(
        self,
        engine: Any,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> np.ndarray:
        distances = engine.batchDistanceOnSP(
            self._matlab_row_vector(p1[:, 0]),
            self._matlab_row_vector(p1[:, 1]),
            self._matlab_row_vector(p2[:, 0]),
            self._matlab_row_vector(p2[:, 1]),
        )
        return self._matlab_numeric_vector_to_numpy(distances)

    def _deduplicated_pair_distances(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> np.ndarray:
        if len(p1) == 0:
            return np.empty(0, dtype=np.float64)

        segment_keys = self._canonical_segment_keys(p1, p2)
        _, unique_indices, inverse = np.unique(
            segment_keys,
            axis=0,
            return_index=True,
            return_inverse=True,
        )
        unique_distances = self.pair_distances(p1[unique_indices], p2[unique_indices])
        return unique_distances[inverse]

    @staticmethod
    def _canonical_segment_keys(
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> np.ndarray:
        swap_mask = (p1[:, 0] > p2[:, 0]) | (
            (p1[:, 0] == p2[:, 0]) & (p1[:, 1] > p2[:, 1])
        )
        starts = p1.copy()
        ends = p2.copy()
        if np.any(swap_mask):
            starts[swap_mask] = p2[swap_mask]
            ends[swap_mask] = p1[swap_mask]
        return np.concatenate((starts, ends), axis=1)

    @staticmethod
    def _matlab_row_vector(values: np.ndarray) -> Any:
        row = np.ascontiguousarray(values, dtype=np.float64).reshape(1, -1).tolist()
        try:
            import matlab
        except ImportError:
            return row
        return matlab.double(row)

    @staticmethod
    def _matlab_numeric_vector_to_numpy(values: Any) -> np.ndarray:
        try:
            return np.asarray(values, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            raw_values = getattr(values, "_data", values)
            return np.asarray(raw_values, dtype=np.float64).reshape(-1)

    @staticmethod
    def _spherical_triangle_area_from_side_lengths(
        side_a: np.ndarray,
        side_b: np.ndarray,
        side_c: np.ndarray,
    ) -> np.ndarray:
        angular_a = np.clip(side_a / MATLAB_SPHERE_RADIUS_UM, 0.0, np.pi)
        angular_b = np.clip(side_b / MATLAB_SPHERE_RADIUS_UM, 0.0, np.pi)
        angular_c = np.clip(side_c / MATLAB_SPHERE_RADIUS_UM, 0.0, np.pi)
        semiperimeter = 0.5 * (angular_a + angular_b + angular_c)

        with np.errstate(divide="ignore", invalid="ignore"):
            tan_term = (
                np.tan(0.5 * semiperimeter)
                * np.tan(0.5 * (semiperimeter - angular_a))
                * np.tan(0.5 * (semiperimeter - angular_b))
                * np.tan(0.5 * (semiperimeter - angular_c))
            )
        tan_term = np.maximum(tan_term, 0.0)
        spherical_excess = 4.0 * np.arctan(np.sqrt(tan_term))
        return spherical_excess * (MATLAB_SPHERE_RADIUS_UM ** 2)

    def _matlab_context(self) -> tuple[Any, str]:
        definitions = import_module("definitions")
        engine = definitions.get_matlab_engine()
        bin_dir = str(getattr(definitions, "BIN_DIR", ""))
        return engine, bin_dir


GEOMETRY_METRIC_DEPENDENCIES: dict[GeometryCapability, tuple[str, ...]] = {
    GeometryCapability.DISTANCE: (
        "disc_fovea_distance_um",
        "disc_diameter_um",
        "disc_major_axis_um",
        "disc_minor_axis_um",
        "vessel_width_um",
        "vessel_width_gradient_um",
        "vessel_width_intercept_um",
        "a_width_um",
        "a_width_gradient_um",
        "a_width_intercept_um",
        "v_width_um",
        "v_width_gradient_um",
        "v_width_intercept_um",
    ),
    GeometryCapability.AREA: (
        "roi_area_mm2",
        "disc_area_mm2",
        "vessel_area_mm2",
        "vessel_density_area",
        "a_area_mm2",
        "a_density_area",
        "v_area_mm2",
        "v_density_area",
    ),
}


def capabilities_for_metrics(metric_names: Iterable[str]) -> tuple[GeometryCapability, ...]:
    metric_name_set = set(metric_names)
    return tuple(
        capability
        for capability, dependent_metrics in GEOMETRY_METRIC_DEPENDENCIES.items()
        if metric_name_set.intersection(dependent_metrics)
    )


def missing_geometry_metrics(capabilities: GeometryCapabilities) -> tuple[str, ...]:
    missing_metrics: list[str] = []
    for capability, dependent_metrics in GEOMETRY_METRIC_DEPENDENCIES.items():
        if not capabilities.supports(capability):
            missing_metrics.extend(dependent_metrics)
    return tuple(missing_metrics)


DEFAULT_GEOMETRY_BACKENDS: tuple[str, ...] = ("matlab",)


def configured_geometry_backends() -> tuple[str, ...]:
    raw = os.getenv("ULTRALYZER_GEOMETRY_BACKENDS")
    if raw is None:
        return DEFAULT_GEOMETRY_BACKENDS

    names = tuple(
        name.strip().lower()
        for name in raw.split(",")
        if name.strip()
    )
    return names or DEFAULT_GEOMETRY_BACKENDS


def build_default_geometry_backends() -> tuple[GeometryBackend, ...]:
    backends: list[GeometryBackend] = []
    logger = logging.getLogger(__name__)
    for name in configured_geometry_backends():
        if name == "matlab":
            backends.append(MatlabGeometryBackend())
        else:
            logger.warning("Unknown geometry backend '%s' ignored.", name)
    return tuple(backends)


_ADAPTER_CACHE: dict[tuple[str, ...], GeometryAdapter] = {}


def get_default_geometry_adapter() -> GeometryAdapter:
    backend_names = configured_geometry_backends()
    adapter = _ADAPTER_CACHE.get(backend_names)
    if adapter is None:
        adapter = GeometryAdapter(build_default_geometry_backends())
        _ADAPTER_CACHE[backend_names] = adapter
    return adapter


class GeometryAdapter:
    """Resolves geometry computations from an ordered list of backends.

    Backends are probed lazily and queried in order. Each capability is served by the
    first backend that reports support for it. Backend statuses are cached until reset() is called.
    """

    def __init__(
        self,
        backends: Sequence[GeometryBackend],
        logger: logging.Logger | None = None,
    ):
        self._backends = tuple(backends)
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._backend_statuses: tuple[GeometryBackendStatus, ...] | None = None

    def backend_statuses(self) -> tuple[GeometryBackendStatus, ...]:
        if self._backend_statuses is None:
            statuses: list[GeometryBackendStatus] = []
            for backend in self._backends:
                try:
                    statuses.append(backend.probe())
                except Exception as exc:
                    self._logger.warning(
                        "Geometry backend '%s' probe failed: %s",
                        backend.name,
                        exc,
                    )
                    statuses.append(
                        GeometryBackendStatus(
                            name=backend.name,
                            capabilities=GeometryCapabilities(),
                            detail=str(exc),
                        )
                    )
            self._backend_statuses = tuple(statuses)
        return self._backend_statuses

    def reset(self) -> None:
        self._backend_statuses = None

    def readiness(self) -> GeometryReadiness:
        capability_backends = {
            GeometryCapability.DISTANCE: self.backend_name_for(GeometryCapability.DISTANCE),
            GeometryCapability.AREA: self.backend_name_for(GeometryCapability.AREA),
        }
        capabilities = GeometryCapabilities(
            distance=capability_backends[GeometryCapability.DISTANCE] is not None,
            area=capability_backends[GeometryCapability.AREA] is not None,
        )
        if capabilities.is_empty():
            traffic_light = GeometryTrafficLight.RED
        elif capabilities.is_complete():
            traffic_light = GeometryTrafficLight.GREEN
        else:
            traffic_light = GeometryTrafficLight.YELLOW
        return GeometryReadiness(
            traffic_light=traffic_light,
            capabilities=capabilities,
            distance_backend=capability_backends[GeometryCapability.DISTANCE],
            area_backend=capability_backends[GeometryCapability.AREA],
            backend_statuses=self.backend_statuses(),
        )

    def supports(self, capability: GeometryCapability) -> bool:
        return self.backend_for(capability) is not None

    def backend_name_for(self, capability: GeometryCapability) -> str | None:
        backend = self.backend_for(capability)
        return None if backend is None else backend.name

    def backend_for(self, capability: GeometryCapability) -> GeometryBackend | None:
        statuses = self.backend_statuses()
        for backend, status in zip(self._backends, statuses):
            if status.capabilities.supports(capability):
                return backend
        return None

    def pair_distances(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        backend = self.backend_for(GeometryCapability.DISTANCE)
        if backend is None:
            raise GeometryCapabilityUnavailableError(
                "No geometry backend provides geodesic distance support."
            )
        return backend.pair_distances(p1, p2)

    def roi_area(
        self,
        points: np.ndarray,
        area_algorithm: str = "cross_product",
    ) -> float:
        backend = self.backend_for(GeometryCapability.AREA)
        if backend is None:
            raise GeometryCapabilityUnavailableError(
                "No geometry backend provides ROI area support."
            )
        return backend.roi_area(points, area_algorithm=area_algorithm)