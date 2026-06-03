from sqlalchemy import create_engine, Column, String, DateTime, Integer, Enum, ForeignKey, Float, Boolean, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, joinedload
from definitions import DB_DIR, IMAGE_FORMATS
from pathlib import Path
from typing import Any
import datetime as dt
import pandas as pd
import enum
import json
import os

Base = declarative_base()

DEFAULT_ROI_DEFINITIONS = (
    {
        "code": "full",
        "name": "Full image",
        "generation_mode": "computed_full",
        "params": {"mask": "lightness_nonzero_largest_blob", "close_kernel_px": 11},
    },
    {
        "code": "central",
        "name": "Central retina",
        "generation_mode": "computed_central",
        "params": {"center": "optic_disc", "radius_px": 300},
    },
    {
        "code": "mid_periphery",
        "name": "Mid-periphery",
        "generation_mode": "template_mask",
        "params": {
            "left_template": ".roi/mid-periphery/left.png",
            "right_template": ".roi/mid-periphery/right.png",
        },
    },
)

LANDMARK_CONTROL_COLUMNS = {"image_id", "name", "algorithm_version", "timestamp", 
                            "crae_px", "crve_px", "avr_px", "crae_um", "crve_um", "avr_um"}
AV_LANDMARK_CONTROL_COLUMNS = {"image_id", "name", "algorithm_version", "timestamp", 
                               "laterality", "disc_center_x", "disc_center_y", "disc_diameter_px", "disc_diameter_um",
                               "disc_area_px", "disc_area_mm2", "disc_major_axis_px", "disc_major_axis_um", "disc_minor_axis_px", "disc_minor_axis_um", 
                               "disc_orientation_deg", "disc_circularity", "disc_eccentricity", "fovea_center_x", "fovea_center_y", 
                               "disc_fovea_distance_px", "disc_fovea_distance_um", "disc_fovea_angle_deg"}
ROI_CONTROL_COLUMNS = {"id", "image_roi_id", "image_id", "algorithm_version", "timestamp"}

class QCDecisionEnum(str, enum.Enum):
    """Enum for QC decisions"""
    PASS = "pass"
    BORDERLINE = "borderline"
    REJECT = "reject"


class MetaData(Base):
    """Metadata of images and processing"""
    __tablename__ = "metadata"
    
    id = Column(Integer, primary_key=True)
    extension = Column(String, nullable=False)
    name = Column(String, unique=True, nullable=False)
    folder = Column(String, nullable=False)


class QCResult(Base):
    """Quality Control result for an image"""
    __tablename__ = "QC"

    id = Column(Integer, ForeignKey("metadata.id"), primary_key=True, unique=True)
    name = Column(String, ForeignKey("metadata.name"), unique=True, nullable=False)
    decision = Column(Enum(QCDecisionEnum), nullable=False)
    notes = Column(String, default="")
    
    timestamp = Column(DateTime, default=dt.datetime.now(dt.timezone.utc))
    
    # Relationship
    meta = relationship("MetaData", foreign_keys=[id])

    def __repr__(self):
        return f"<QCResult(name='{self.name}', decision='{self.decision}')>"


class SegmentationResult(Base):
    """Segmentation result for an image"""
    __tablename__ = "segmentation"

    id = Column(Integer, ForeignKey("metadata.id"), primary_key=True)
    extension = Column(String, nullable=False)
    name = Column(String, ForeignKey("metadata.name"),
                  unique=True, nullable=False)
    
    # Mask path
    seg_folder = Column(String, nullable=False)
    
    # Metadata
    model_name = Column(String, default="dummy")
    model_version = Column(String, default="1.0")
    
    timestamp = Column(DateTime, default=dt.datetime.now(dt.timezone.utc))

    # Relationship
    meta = relationship("MetaData", foreign_keys=[id])
    
    def __repr__(self):
        return f"<SegmentationResult(name={self.name + self.extension}, seg_folder={self.seg_folder})>"
    

class LandmarkMetricsMixin:
    """Shared landmark metrics stored once per image."""

    laterality = Column(String, nullable=True)
    # OPTIC DISC
    disc_center_x = Column(Float, nullable=True)
    disc_center_y = Column(Float, nullable=True)
    disc_diameter_px = Column(Float, nullable=True)
    disc_diameter_um = Column(Float, nullable=True)
    disc_area_px = Column(Float, nullable=True)
    disc_area_mm2 = Column(Float, nullable=True)
    disc_major_axis_px = Column(Float, nullable=True)
    disc_major_axis_um = Column(Float, nullable=True)
    disc_minor_axis_px = Column(Float, nullable=True)
    disc_minor_axis_um = Column(Float, nullable=True)
    disc_orientation_deg = Column(Float, nullable=True)
    disc_circularity = Column(Float, nullable=True)
    disc_eccentricity = Column(Float, nullable=True)
    # FOVEA
    fovea_center_x = Column(Float, nullable=True)
    fovea_center_y = Column(Float, nullable=True)
    # OPTIC DISC - FOVEA RELATIONSHIP
    disc_fovea_distance_px = Column(Float, nullable=True)
    disc_fovea_distance_um = Column(Float, nullable=True)
    disc_fovea_angle_deg = Column(Float, nullable=True)
    # ARTEY - VEIN LANDMARKS
    crae_px = Column(Float, nullable=True)
    crae_um = Column(Float, nullable=True)
    crve_px = Column(Float, nullable=True)
    crve_um = Column(Float, nullable=True)
    avr_px = Column(Float, nullable=True)
    avr_um = Column(Float, nullable=True)


class ROIMetricsMixin:
    """Shared metrics that depend on selected ROI."""

    # VESSELS
    vessel_density = Column(Float, nullable=True)
    vessel_density_area = Column(Float, nullable=True)
    vessel_tortuosity_density = Column(Float, nullable=True)
    vessel_tortuosity_fft = Column(Float, nullable=True)
    vessel_fractal_dimension_sandbox = Column(Float, nullable=True)
    vessel_fractal_dimension_boxcount = Column(Float, nullable=True)
    vessel_area_mm2 = Column(Float, nullable=True)
    vessel_width_px = Column(Float, nullable=True)
    vessel_width_um = Column(Float, nullable=True)
    vessel_width_gradient_px = Column(Float, nullable=True)
    vessel_width_gradient_um = Column(Float, nullable=True)
    vessel_width_intercept_px = Column(Float, nullable=True)
    vessel_width_intercept_um = Column(Float, nullable=True)
    # ARTERIES
    a_density = Column(Float, nullable=True)
    a_density_area = Column(Float, nullable=True)
    a_tortuosity_density = Column(Float, nullable=True)
    a_tortuosity_fft = Column(Float, nullable=True)
    a_fractal_dimension_sandbox = Column(Float, nullable=True)
    a_fractal_dimension_boxcount = Column(Float, nullable=True)
    a_area_mm2 = Column(Float, nullable=True)
    a_width_px = Column(Float, nullable=True)
    a_width_um = Column(Float, nullable=True)
    a_width_gradient_px = Column(Float, nullable=True)
    a_width_gradient_um = Column(Float, nullable=True)
    a_width_intercept_px = Column(Float, nullable=True)
    a_width_intercept_um = Column(Float, nullable=True)
    a_groups = Column(Float, nullable=True)
    a_branching_points = Column(Float, nullable=True)
    a_branches = Column(Float, nullable=True)
    # VEINS
    v_density = Column(Float, nullable=True)
    v_density_area = Column(Float, nullable=True)
    v_tortuosity_density = Column(Float, nullable=True)
    v_tortuosity_fft = Column(Float, nullable=True)
    v_fractal_dimension_sandbox = Column(Float, nullable=True)
    v_fractal_dimension_boxcount = Column(Float, nullable=True)
    v_area_mm2 = Column(Float, nullable=True)
    v_width_px = Column(Float, nullable=True)
    v_width_um = Column(Float, nullable=True)
    v_width_gradient_px = Column(Float, nullable=True)
    v_width_gradient_um = Column(Float, nullable=True)
    v_width_intercept_px = Column(Float, nullable=True)
    v_width_intercept_um = Column(Float, nullable=True)
    v_groups = Column(Float, nullable=True)
    v_branching_points = Column(Float, nullable=True)
    v_branches = Column(Float, nullable=True)
    # ARTERIES - VEINS RELATIONSHIP
    av_ratio_px = Column(Float, nullable=True)
    av_ratio_um = Column(Float, nullable=True)
    av_crossings = Column(Float, nullable=True)
    av_arcade_concavity = Column(Float, nullable=True)


class ROIDefinition(Base):
    """Catalog of supported ROI types."""
    __tablename__ = "roi_definition"

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    generation_mode = Column(String, nullable=False)
    params_json = Column(Text, nullable=True)
    active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc),
                        onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    image_rois = relationship("ImageROI", back_populates="roi_definition")

    def __repr__(self):
        return f"<ROIDefinition(code={self.code}, generation_mode={self.generation_mode})>"


class ImageROI(Base):
    """Concrete ROI instance resolved for one image."""
    __tablename__ = "image_roi"
    __table_args__ = (
        UniqueConstraint("image_id", "roi_definition_id", name="uq_image_roi_definition"),
    )

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("metadata.id"), nullable=False, index=True)
    roi_definition_id = Column(Integer, ForeignKey("roi_definition.id"), nullable=False, index=True)
    selected_for_metrics = Column(Boolean, default=False, nullable=False)
    mask_path = Column(String, nullable=True)
    geometry_json = Column(Text, nullable=True)
    area_px = Column(Float, nullable=True)
    area_mm2 = Column(Float, nullable=True)

    created_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc),
                        onupdate=lambda: dt.datetime.now(dt.timezone.utc))

    meta = relationship("MetaData", foreign_keys=[image_id])
    roi_definition = relationship("ROIDefinition", back_populates="image_rois", foreign_keys=[roi_definition_id])
    roi_metrics = relationship("ROIMetricsResult", back_populates="image_roi")

    def __repr__(self):
        return f"<ImageROI(image_id={self.image_id}, roi_definition_id={self.roi_definition_id})>"


class LandmarkMetricsResult(LandmarkMetricsMixin, Base):
    """Landmark metrics stored once per image, independent of ROI choice."""
    __tablename__ = "metrics_landmark"

    image_id = Column(Integer, ForeignKey("metadata.id"), primary_key=True)
    name = Column(String, ForeignKey("metadata.name"), unique=True, nullable=False)
    algorithm_version = Column(String, default="1.0", nullable=False)
    timestamp = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))

    meta = relationship("MetaData", foreign_keys=[image_id])

    def __repr__(self):
        return f"<LandmarkMetricsResult(name={self.name})>"


class ROIMetricsResult(ROIMetricsMixin, Base):
    """Metrics stored for one ROI instance on one image."""
    __tablename__ = "metrics_roi"
    __table_args__ = (
        UniqueConstraint("image_roi_id", "algorithm_version", name="uq_metrics_roi_version"),
    )

    id = Column(Integer, primary_key=True)
    image_roi_id = Column(Integer, ForeignKey("image_roi.id"), nullable=False, index=True)
    image_id = Column(Integer, ForeignKey("metadata.id"), nullable=False, index=True)
    algorithm_version = Column(String, default="1.0", nullable=False)
    timestamp = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc))

    image_roi = relationship("ImageROI", back_populates="roi_metrics", foreign_keys=[image_roi_id])
    meta = relationship("MetaData", foreign_keys=[image_id])

    def __repr__(self):
        return f"<ROIMetricsResult(image_id={self.image_id}, image_roi_id={self.image_roi_id})>"


class DatabaseManager:
    """Manages database connection and operations"""
    
    def __init__(self, db_path: Path = None):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to database file. If None, uses ':memory:'
        """
        if db_path is None:
            self.db_path = Path(os.path.join(DB_DIR, "ultralyzer.db"))
        else:
            self.db_path = Path(db_path)
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create engine
        db_url = f"sqlite:///{self.db_path}"
        self.engine = create_engine(db_url, echo=False)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.seed_roi_definitions()

    def _metric_values_for_model(
        self,
        model: Any,
        values: dict,
        excluded_columns: set[str]) -> dict:
        """Filter a metrics dict to columns accepted by a model."""
        metric_columns = {
            column.name for column in model.__table__.columns
            if column.name not in excluded_columns
        }
        return {
            key: value for key, value in values.items()
            if key in metric_columns
        }
    
    ############ PROPERTIES ############
    
    @property
    def session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    ############ METADATA GET METHODS ############
    
    def get_metadata_by_filename(self, name: str) -> MetaData:
        """Get metadata for a specific image by filename"""
        session = self.session
        if name.endswith(tuple(IMAGE_FORMATS)):
            name = name.rsplit('.', 1)[0]
        try:
            meta = session.query(MetaData).filter_by(name=name).first()
            return meta
        finally:
            session.close()
    
    ############ METADATA SET METHODS ############
    
    def save_folder_metadata(self, folder: Path) -> bool:
        """
        Save metadata for all images in a folder.
        
        Args:
            folder: Path to image folder
            
        Returns:
            True if successful, False otherwise
        """
        session = self.session
        try:
            image_files = [
                f for f in folder.iterdir()
                if f.suffix.lower() in IMAGE_FORMATS
            ]
            image_files.sort()
            
            for img_path in image_files:
                name = str(img_path.stem)
                extension = str(img_path.suffix.lower())
                folder_str = str(folder)
                
                # Check if metadata already exists
                existing = session.query(MetaData).filter_by(name=name).first()
                if not existing:
                    meta = MetaData(
                        extension=extension,
                        name=name,
                        folder=folder_str
                    )
                    session.add(meta)
                elif str(existing.folder) != folder_str:
                    existing.folder = folder_str
                    
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"Error saving folder metadata: {str(e)}")
            return False
        
        finally:
            session.close()
    
    def save_image_metadata(self, name: str, extension: str, folder: Path) -> bool:
        """
        Save metadata for a single image.
        
        Args:
            name: Name of the image file without extension
            extension: Extension of the image file
            folder: Path to image folder
        Returns:
            True if successful, False otherwise
        """
        session = self.session
        try:
            if extension.lower() in IMAGE_FORMATS:
                extension = extension.lower()
                name = str(Path(name).stem)
            else:
                return False
            folder_str = str(folder)
            
            # Check if metadata already exists
            existing = session.query(MetaData).filter_by(name=name).first()
            if not existing:
                meta = MetaData(
                    extension=extension,
                    name=name,
                    folder=folder_str
                )
                session.add(meta)
            elif str(existing.folder) != folder_str:
                existing.folder = folder_str
                
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"Error saving image metadata: {str(e)}")
            return False
        
        finally:
            session.close()
    
    ############ QC GET METHODS ############
    
    def get_qc_result(self, name: str) -> QCResult:
        """Get QC result for a specific image"""
        session = self.session
        try:
            result = session.query(QCResult).filter_by(name=name).first()
            return result
        finally:
            session.close()
    
    ############ QC SET METHODS ############

    def save_qc_result(self, name: str, decision: str, notes: str = "") -> bool:
        """
        Save or update QC result for an image.
        
        Args:
            name: Name of the image
            decision: Decision (pass, borderline, reject)
            notes: Optional notes about the image
            
        Returns:
            True if successful, False otherwise
        """
        session = self.session
        try:
            # Check if metadata exists
            meta = session.query(MetaData).filter_by(name=name).first()
            if not meta:
                print(f"Error: No metadata found for {name}")
                return False
            
            # Check if QC result exists
            existing = session.query(QCResult).filter_by(name=name).first()

            if existing:
                # Update existing
                existing.decision = QCDecisionEnum(decision)
                existing.notes = notes
                existing.timestamp = dt.datetime.now(dt.timezone.utc)
            else:
                # Create new
                qc_result = QCResult(
                    id=meta.id,
                    name=name,
                    decision=QCDecisionEnum(decision),
                    notes=notes
                )
                session.add(qc_result)
            
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"Error saving QC result: {str(e)}")
            return False
        
        finally:
            session.close()
    
    ############ SEGMENTATION GET METHODS ############
    
    def get_segmentation_mask_path(self, name: str) -> Path:
        """Get segmentation mask path for a specific image"""
        session = self.session
        try:
            result = session.query(SegmentationResult).filter_by(
                name=name
            ).first()
            if result:
                return Path(result.seg_folder)
            else:
                return None
        finally:
            session.close()
    
    def get_pending_segmentations(self) -> list:
        """Get QC results that need segmentation (PASS or BORDERLINE without segmentation)"""
        session = self.session
        results = []
        try:
            results = session.query(MetaData).join(
                QCResult, MetaData.id == QCResult.id
            ).filter(
                QCResult.decision.in_([QCDecisionEnum.PASS, QCDecisionEnum.BORDERLINE])
            ).filter(
                ~QCResult.id.in_(
                    session.query(SegmentationResult.id)
                )
            ).all()
            return results
        finally:
            session.close()
    
    def get_segmentation_by_filename(self, name: str) -> SegmentationResult:
        """Get segmentation result for a specific image"""
        session = self.session
        try:
            result = session.query(SegmentationResult).filter_by(name=name).first()
            return result
        finally:
            session.close()
    
    def get_segmentation_result_by_id(self, id: int):
        """Get segmentation result for a QC result"""
        session = self.session
        try:
            result = session.query(SegmentationResult).filter_by(id=id).first()
            return result
        finally:
            session.close()
    
    ############ SEGMENTATION SET METHODS ############

    def set_mask_info(self, id: int, mask_path: Path, suffix: Path) -> bool:
        """
        Set mask information for an image.
        
        Args:
            id: ID of the image metadata
            mask_path: Path to the mask file
            mask_type: Type of mask ('av' or 'vessel')
        Returns:
            True if successful, False otherwise
        """
        session = self.session
        try:
            meta = session.query(MetaData).filter_by(id=id).first()
            if not meta:
                print(f"Error: No metadata found for ID {id}")
                return False
            
            seg_result = session.query(SegmentationResult).filter_by(id=id).first()
            # Add new entry if not exists
            if not seg_result:
                seg_result = SegmentationResult(
                    id=id,
                    extension=str(suffix).lower(),
                    name=meta.name,
                    seg_folder=""
                )
                session.add(seg_result)
            
            # Update (now) existing entry
            seg_result.seg_folder = str(mask_path)
            
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"Error setting mask info: {str(e)}")
            return False
        
        finally:
            session.close()
    
    def save_segmentation_result(
        self,
        id: int,
        extension: str,
        seg_folder: str,
        model_name: str = "default_model",
        model_version: str = "1.0") -> bool:
        """
        Save segmentation result for an image.
        
        Args:
            qc_result_id: ID of the associated QC result
            seg_folder: Path to segmentation mask folder
            model_name: Name of the segmentation model
            model_version: Version of the segmentation model
            
        Returns:
            True if successful, False otherwise
        """
        session = self.session
        try:
            meta = session.query(MetaData).filter_by(id=id).first()
            if not meta:
                print(f"Error: No metadata found for ID {id}")
                return False

            # Check if segmentation already exists for this QC result
            existing = session.query(SegmentationResult).filter_by(id=id).first()
            
            if existing:
                # Update existing
                existing.seg_folder = seg_folder
                existing.model_name = model_name
                existing.model_version = model_version
                existing.timestamp = dt.datetime.now(dt.timezone.utc)
            else:
                # Create new
                seg_result = SegmentationResult(
                    id=id,
                    extension=extension,
                    name=meta.name,
                    seg_folder=seg_folder,
                    model_name=model_name,
                    model_version=model_version
                )
                session.add(seg_result)
            
            session.commit()
            return True
        
        except Exception as e:
            session.rollback()
            print(f"Error saving segmentation result: {str(e)}")
            return False
        
        finally:
            session.close()
            
            
    ############ ROI GET/SET METHODS ############

    def seed_roi_definitions(self) -> bool:
        """Create or update built-in ROI definitions."""
        session = self.session
        try:
            for definition in DEFAULT_ROI_DEFINITIONS:
                params_json = json.dumps(definition["params"], sort_keys=True)
                existing = session.query(ROIDefinition).filter_by(
                    code=definition["code"]
                ).first()

                if existing:
                    existing.name = definition["name"]
                    existing.generation_mode = definition["generation_mode"]
                    existing.params_json = params_json
                    existing.active = True
                    existing.updated_at = dt.datetime.now(dt.timezone.utc)
                else:
                    session.add(ROIDefinition(
                        code=definition["code"],
                        name=definition["name"],
                        generation_mode=definition["generation_mode"],
                        params_json=params_json,
                        active=True,
                    ))

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error seeding ROI definitions: {str(e)}")
            return False
        finally:
            session.close()

    def get_roi_definitions(self, active_only: bool = True) -> list[ROIDefinition]:
        """Get available ROI definitions."""
        session = self.session
        try:
            query = session.query(ROIDefinition)
            if active_only:
                query = query.filter_by(active=True)
            return query.order_by(ROIDefinition.id).all()
        finally:
            session.close()

    def ensure_image_roi(
        self,
        image_id: int,
        roi_code: str,
        selected_for_metrics: bool | None = None,
        mask_path: str | None = None,
        geometry_json: str | None = None,
        area_px: float | None = None,
        area_mm2: float | None = None) -> int | None:
        """Create or update the concrete ROI row for one image."""
        session = self.session
        try:
            roi_definition = session.query(ROIDefinition).filter_by(code=roi_code).first()
            if not roi_definition:
                print(f"Error: No ROI definition found for code '{roi_code}'")
                return None

            meta = session.query(MetaData).filter_by(id=image_id).first()
            if not meta:
                print(f"Error: No metadata found for ID {image_id}")
                return None

            image_roi = session.query(ImageROI).filter_by(
                image_id=image_id,
                roi_definition_id=roi_definition.id
            ).first()

            if not image_roi:
                image_roi = ImageROI(
                    image_id=image_id,
                    roi_definition_id=roi_definition.id,
                    selected_for_metrics=bool(selected_for_metrics),
                    mask_path=mask_path,
                    geometry_json=geometry_json,
                    area_px=area_px,
                    area_mm2=area_mm2,
                )
                session.add(image_roi)
            else:
                if selected_for_metrics is not None:
                    image_roi.selected_for_metrics = selected_for_metrics
                if mask_path is not None:
                    image_roi.mask_path = mask_path
                if geometry_json is not None:
                    image_roi.geometry_json = geometry_json
                if area_px is not None:
                    image_roi.area_px = area_px
                if area_mm2 is not None:
                    image_roi.area_mm2 = area_mm2
                image_roi.updated_at = dt.datetime.now(dt.timezone.utc)

            session.commit()
            return int(image_roi.id)
        except Exception as e:
            session.rollback()
            print(f"Error ensuring image ROI: {str(e)}")
            return None
        finally:
            session.close()

    def get_image_rois_by_image_id(
        self,
        image_id: int,
        selected_only: bool = False) -> list[ImageROI]:
        """Get ROI instances for one image."""
        session = self.session
        try:
            query = session.query(ImageROI).options(
                joinedload(ImageROI.roi_definition)
            ).filter_by(image_id=image_id)
            if selected_only:
                query = query.filter_by(selected_for_metrics=True)
            return query.order_by(ImageROI.id).all()
        finally:
            session.close()

    def select_image_roi(self, image_id: int, roi_code: str, selected: bool = True) -> bool:
        """Mark an image ROI as selected or unselected for metric computation."""
        image_roi_id = self.ensure_image_roi(
            image_id=image_id,
            roi_code=roi_code,
            selected_for_metrics=selected
        )
        return image_roi_id is not None

    ############ LANDMARK / ROI METRICS METHODS ############

    def save_landmark_metrics_by_id(
        self,
        image_id: int,
        metrics: dict,
        algorithm_version: str = "1.0") -> bool:
        """Save or update image-level landmark metrics."""
        metric_values = self._metric_values_for_model(
            LandmarkMetricsResult,
            metrics,
            LANDMARK_CONTROL_COLUMNS
        )
        if not metric_values:
            return True

        session = self.session
        try:
            meta = session.query(MetaData).filter_by(id=image_id).first()
            if not meta:
                print(f"Error: No metadata found for ID {image_id}")
                return False

            existing = session.query(LandmarkMetricsResult).filter_by(image_id=image_id).first()
            if existing:
                for key, value in metric_values.items():
                    setattr(existing, key, value)
                existing.algorithm_version = algorithm_version
                existing.timestamp = dt.datetime.now(dt.timezone.utc)
            else:
                session.add(LandmarkMetricsResult(
                    image_id=image_id,
                    name=meta.name,
                    algorithm_version=algorithm_version,
                    **metric_values,
                ))

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error saving landmark metrics: {str(e)}")
            return False
        finally:
            session.close()

    def save_landmark_metrics_by_name(
        self,
        name: str,
        metrics: dict,
        algorithm_version: str = "1.0") -> bool:
        """Save or update landmark metrics using image name."""
        meta = self.get_metadata_by_filename(name)
        if not meta:
            print(f"Error: No metadata found for name {name}")
            return False
        return self.save_landmark_metrics_by_id(meta.id, metrics, algorithm_version)

    def get_landmark_metrics_by_filename(self, name: str) -> LandmarkMetricsResult | Any:
        """Get image-level landmark metrics by image name."""
        session = self.session
        try:
            return session.query(LandmarkMetricsResult).filter_by(name=name).first()
        finally:
            session.close()

    def save_av_landmark_metrics_by_id(
        self,
        image_id: int,
        metrics: dict,
        algorithm_version: str = "1.0") -> bool:
        """Save or update CRAE/CRVE/AVR landmark metrics."""
        metric_values = self._metric_values_for_model(
            LandmarkMetricsResult,
            metrics,
            AV_LANDMARK_CONTROL_COLUMNS
        )
        if not metric_values:
            return True

        session = self.session
        try:
            meta = session.query(MetaData).filter_by(id=image_id).first()
            if not meta:
                print(f"Error: No metadata found for ID {image_id}")
                return False

            existing = session.query(LandmarkMetricsResult).filter_by(image_id=image_id).first()
            if existing:
                for key, value in metric_values.items():
                    setattr(existing, key, value)
                existing.algorithm_version = algorithm_version
                existing.timestamp = dt.datetime.now(dt.timezone.utc)
            else:
                session.add(LandmarkMetricsResult(
                    image_id=image_id,
                    name=meta.name,
                    algorithm_version=algorithm_version,
                    **metric_values,
                ))

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error saving landmark metrics: {str(e)}")
            return False
        finally:
            session.close()
    
    def save_roi_metrics(
        self,
        image_roi_id: int,
        metrics: dict,
        algorithm_version: str = "1.0") -> bool:
        """Save or update metrics for one concrete image ROI."""
        metric_values = self._metric_values_for_model(
            ROIMetricsResult,
            metrics,
            ROI_CONTROL_COLUMNS
        )
        if not metric_values:
            return True

        session = self.session
        try:
            image_roi = session.query(ImageROI).filter_by(id=image_roi_id).first()
            if not image_roi:
                print(f"Error: No image ROI found for ID {image_roi_id}")
                return False

            existing = session.query(ROIMetricsResult).filter_by(
                image_roi_id=image_roi_id,
                algorithm_version=algorithm_version
            ).first()

            if existing:
                for key, value in metric_values.items():
                    setattr(existing, key, value)
                existing.timestamp = dt.datetime.now(dt.timezone.utc)
            else:
                session.add(ROIMetricsResult(
                    image_roi_id=image_roi_id,
                    image_id=image_roi.image_id,
                    algorithm_version=algorithm_version,
                    **metric_values,
                ))

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            print(f"Error saving ROI metrics: {str(e)}")
            return False
        finally:
            session.close()

    def has_roi_metrics(
        self,
        image_roi_id: int,
        algorithm_version: str = "1.0") -> bool:
        """Return whether metrics already exist for one concrete image ROI."""
        session = self.session
        try:
            return session.query(ROIMetricsResult.id).filter(
                ROIMetricsResult.image_roi_id == image_roi_id,
                ROIMetricsResult.algorithm_version == algorithm_version,
            ).first() is not None
        finally:
            session.close()

    ############ METRICS GET METHODS ############
    
    def get_metrics_by_filename(self, name: str) -> LandmarkMetricsResult | Any:
        """Get image-level landmark metrics for a specific image."""
        return self.get_landmark_metrics_by_filename(name)
    
    def get_fovea_by_filename(self, name: str) -> tuple[float, float] | Any:
        """Get fovea metrics for a specific image"""
        result = self.get_landmark_metrics_by_filename(name)
        return (result.fovea_center_x, result.fovea_center_y) if result else (None, None)
    
    def get_pending_metrics(self) -> list:
        """Get QC results that need segmentation (PASS or BORDERLINE)"""
        session = self.session
        results = []
        try:
            results = session.query(MetaData).join(
                QCResult, MetaData.id == QCResult.id
            ).filter(
                QCResult.decision.in_([QCDecisionEnum.PASS, QCDecisionEnum.BORDERLINE])
            ).all()
        finally:
            session.close()
            return results
    
    ############ METRICS SET METHODS ############

    def save_metrics_disc_centroid_by_id(
        self,
        id: int,
        disc_x: float,
        disc_y: float) -> bool:
        """
        Save optic disc centroid metrics for an image.
        
        Args:
            id: ID of the associated metadata
            disc_x: Optic disc center x coordinate
            disc_y: Optic disc center y coordinate
        Returns:
            True if successful, False otherwise
        """
        return self.save_landmark_metrics_by_id(id, {
            "disc_center_x": disc_x,
            "disc_center_y": disc_y,
        })
        
    def save_metrics_fovea_by_id(
        self,
        id: int,
        fovea_x: float,
        fovea_y: float) -> bool:
        """
        Save fovea metrics for an image.
        
        Args:
            id: ID of the associated metadata
            fovea_x: Fovea center x coordinate
            fovea_y: Fovea center y coordinate
        Returns:
            True if successful, False otherwise
        """
        return self.save_landmark_metrics_by_id(id, {
            "fovea_center_x": fovea_x,
            "fovea_center_y": fovea_y,
        })
    
    def save_metrics_fovea_by_name(
        self,
        name: str,
        fovea_x: float,
        fovea_y: float) -> bool:
        """
        Save fovea metrics for an image by name.
        
        Args:
            name: Name of the image
            fovea_x: Fovea center x coordinate
            fovea_y: Fovea center y coordinate
        Returns:
            True if successful, False otherwise
        """
        return self.save_landmark_metrics_by_name(name, {
            "fovea_center_x": fovea_x,
            "fovea_center_y": fovea_y,
        })
    
    def save_metrics_laterality_by_id(
        self,
        id: int,
        laterality: str) -> bool:
        """
        Save laterality metric for an image.
        
        Args:
            id: ID of the associated metadata
            laterality: Laterality value ('left' or 'right')
        Returns:
            True if successful, False otherwise
        """
        # Check if laterality is valid
        if laterality not in ['left', 'right']:
            print(f"Error: Invalid laterality value '{laterality}' for ID {id}")
            return False
        
        return self.save_landmark_metrics_by_id(id, {"laterality": laterality})
    
    ############ METRICS EXPORT METHODS ############

    def _excel_path(self, save_path: Path) -> Path:
        save_path = Path(save_path)
        if save_path.suffix.lower() not in {".xlsx", ".xlsm"}:
            return save_path.with_suffix(".xlsx")
        return save_path

    def _safe_excel_sheet_name(self, name: str, used_names: set[str]) -> str:
        clean_name = str(name or "Sheet")
        for char in r"[]:*?/\\":
            clean_name = clean_name.replace(char, "_")
        clean_name = clean_name.strip() or "Sheet"
        clean_name = clean_name[:31]

        candidate = clean_name
        counter = 1
        while candidate in used_names:
            suffix = f"_{counter}"
            candidate = f"{clean_name[:31 - len(suffix)]}{suffix}"
            counter += 1

        used_names.add(candidate)
        return candidate

    def _export_filename(self, meta: MetaData) -> str:
        return f"{meta.name}{meta.extension}"

    def _qc_results_dataframe(self, session) -> pd.DataFrame:
        results = session.query(
            QCResult,
            MetaData,
        ).join(
            MetaData, QCResult.id == MetaData.id
        ).order_by(MetaData.name).all()
        data = [{
            "filename": self._export_filename(meta),
            "result": qc_result.decision.value if qc_result.decision else None,
            "notes": qc_result.notes,
        } for qc_result, meta in results]
        return pd.DataFrame(data, columns=["filename", "result", "notes"])

    def _landmark_metrics_dataframe(self, session) -> pd.DataFrame:
        results = session.query(
            LandmarkMetricsResult,
            MetaData,
        ).join(
            MetaData, LandmarkMetricsResult.image_id == MetaData.id
        ).order_by(MetaData.name).all()

        metric_columns = [
            column.name for column in LandmarkMetricsResult.__table__.columns
            if column.name not in set(LANDMARK_CONTROL_COLUMNS).intersection(AV_LANDMARK_CONTROL_COLUMNS)
        ]
        base_columns = ["filename"]
        data = []
        for landmark_metrics, meta in results:
            row = {
                "filename": self._export_filename(meta),
            }
            for column in metric_columns:
                row[column] = getattr(landmark_metrics, column)
            data.append(row)

        return pd.DataFrame(data, columns=base_columns + metric_columns)

    def _roi_metrics_dataframes(self, session) -> dict[str, pd.DataFrame]:
        roi_definitions = session.query(ROIDefinition).order_by(ROIDefinition.id).all()
        metric_columns = [
            column.name for column in ROIMetricsResult.__table__.columns
            if column.name not in ROI_CONTROL_COLUMNS
        ]
        base_columns = [
            "filename",
            "roi_area_px",
            "roi_area_mm2",
        ]
        columns = base_columns + metric_columns

        data_by_roi = {str(roi_definition.code): [] for roi_definition in roi_definitions}
        results = session.query(
            ROIMetricsResult,
            ImageROI,
            ROIDefinition,
            MetaData,
        ).join(
            ImageROI, ROIMetricsResult.image_roi_id == ImageROI.id
        ).join(
            ROIDefinition, ImageROI.roi_definition_id == ROIDefinition.id
        ).join(
            MetaData, ROIMetricsResult.image_id == MetaData.id
        ).order_by(
            ROIDefinition.id,
            MetaData.name,
            ROIMetricsResult.algorithm_version,
        ).all()

        for roi_metrics, image_roi, roi_definition, meta in results:
            roi_code = str(roi_definition.code)
            row = {
                "filename": self._export_filename(meta),
                "roi_area_px": image_roi.area_px,
                "roi_area_mm2": image_roi.area_mm2,
            }
            for column in metric_columns:
                row[column] = getattr(roi_metrics, column)
            data_by_roi[roi_code].append(row)

        return {
            roi_code: pd.DataFrame(rows, columns=columns)
            for roi_code, rows in data_by_roi.items()
        }

    def export_results_workbook(self, save_path: Path) -> bool:
        """Export landmark metrics, ROI metric sheets, and QC results to one Excel workbook."""
        session = self.session
        try:
            workbook_path = self._excel_path(save_path)
            used_names = set()

            with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
                landmark_sheet = self._safe_excel_sheet_name("Landmark Metrics", used_names)
                self._landmark_metrics_dataframe(session).to_excel(
                    writer,
                    sheet_name=landmark_sheet,
                    index=False,
                )

                roi_dataframes = self._roi_metrics_dataframes(session)
                for roi_code, dataframe in roi_dataframes.items():
                    sheet_name = self._safe_excel_sheet_name(f"ROI {roi_code}", used_names)
                    dataframe.to_excel(writer, sheet_name=sheet_name, index=False)

                qc_sheet = self._safe_excel_sheet_name("QC", used_names)
                self._qc_results_dataframe(session).to_excel(writer, sheet_name=qc_sheet, index=False)

            return True
        except Exception as e:
            print(f"Error exporting results workbook: {str(e)}")
            return False
        finally:
            session.close()

    def export_metrics_results(self, save_path: Path) -> bool:
        """
        Export all results to one Excel workbook.
        
        Args:
            save_path: Path to save the CSV file
            
        Returns:
            True if successful, False otherwise
        """
        return self.export_results_workbook(save_path)
        
