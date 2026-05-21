import cv2
import numpy as np
from PIL import Image
from typing import Any
from backend.models.database import DatabaseManager
from sklearn.preprocessing import PolynomialFeatures
from skimage.morphology import area_opening, skeletonize
from scipy.ndimage import rotate, distance_transform_edt
from skimage.transform import hough_circle, hough_circle_peaks
from sklearn.linear_model import RANSACRegressor, LinearRegression


class ArcadeDetector(object):
    
    def __init__(self, 
                 name: str, 
                 mask: np.ndarray, 
                 db_manager: DatabaseManager = None,
                 disc_center_x: float = None,
                 disc_center_y: float = None,
                 disc_diameter_px: float = None,
                 disc_fovea_angle_deg: float = None,
                 laterality: str = None):
        self.name: str = name
        self.mask: np.ndarray = mask
        self.mask_original: np.ndarray = mask.copy()
        self.db_manager: DatabaseManager = db_manager or DatabaseManager()
        self._disc_center_x = disc_center_x
        self._disc_center_y = disc_center_y
        self._disc_diameter_px = disc_diameter_px
        self._disc_fovea_angle_deg = disc_fovea_angle_deg
        self._laterality = laterality
        
        # Convert to 2D if image has more than 2 dimensions
        if len(self.mask.shape) > 2:
            self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
    
    def __call__(self):
        
        # Crop mask to disc region
        self.crop_around_disc()
        
        # Rotate mask to vertical orientation
        try:
            self.rotate(-self.angle)
        except Exception as e:
            print(f"Could not rotate image {self.name} - {e}")
        
        # Remove small vessels by distance transform
        self.dist_transform()
        
        # Further remove small vessels by morphological area opening
        self.area_opening()
        
        # Detect parabola via circle Hough transform
        self.detect_parabola()
        
        # Remove small vessels by morphological area opening
        self.area_opening()
        
        # Image reconstruction via morphological closing using a rectangular kernel rotated between -70 & 70
        self.rectangular_closing()
        
        # Skeletonize the mask
        self.skeleton()
    
    @property
    def metrics(self):
        return self.db_manager.get_metrics_by_filename(self.name)
    
    @property
    def angle(self) -> float | Any:
        if self._disc_fovea_angle_deg is not None:
            return self._disc_fovea_angle_deg
        return self.metrics.disc_fovea_angle_deg if self.metrics else None
    
    @property
    def eye(self) -> str | Any:
        if self._laterality is not None:
            return self._laterality
        return self.metrics.laterality if self.metrics else None
    
    @property
    def disc_x(self) -> float | Any:
        if self._disc_center_x is not None:
            return self._disc_center_x
        return self.metrics.disc_center_x if self.metrics else None
    
    @property
    def disc_y(self) -> float | Any:
        if self._disc_center_y is not None:
            return self._disc_center_y
        return self.metrics.disc_center_y if self.metrics else None
    
    @property
    def disc_diameter(self) -> float | Any:
        if self._disc_diameter_px is not None:
            return self._disc_diameter_px
        return self.metrics.disc_diameter_px if self.metrics else None
    
    ############ GETTER/SETTER ############
    
    def get_coordinates(self, mask: np.ndarray = None):
        """Get vessel pixel coordinates from the mask.
        
        Returns:
            coords: Nx2 array with (x, y) coordinates of vessel pixels"""
        if mask is None:
            mask = self.mask
        bool_array = np.where(mask > 0)
        coords = np.flip(np.column_stack(bool_array), axis=1)
        return coords
    
    ############ PREPROCESSING ############          

    def dist_transform(self, min_trigger_area=1000, min_trigger_num=2, threshold_quantile=0.99):
        areas = self._compute_areas()
        if sum(areas > min_trigger_area) >= min_trigger_num:
            dt_mask = np.asarray(distance_transform_edt(self.mask))
            thres = np.quantile(dt_mask, threshold_quantile)
            self.mask = ((dt_mask > thres)*255).astype(np.uint8)

    def area_opening(self, min_trigger_area=1000, min_trigger_num=2, threshold_quantile=0.8, threshold_cap=300, connectivity=1):
        areas = self._compute_areas()
        if sum(areas > min_trigger_area) >= min_trigger_num:
            area_threshold_interim = np.quantile(areas, threshold_quantile)
            area_threshold = min(area_threshold_interim, threshold_cap)
            self.mask = area_opening(self.mask, area_threshold, connectivity=connectivity)
            
    def detect_parabola(self, min_trial_radius=5, max_trial_radius=15, binary_quantile=0.985):
        hough_radii = np.arange(min_trial_radius, max_trial_radius + 1, 1)
        hough_results = hough_circle(self.mask, hough_radii)
        # Select the most prominent circle
        _, _, _, radius = hough_circle_peaks(hough_results, hough_radii, total_num_peaks=1)
        
        self.mask = hough_circle(self.mask, radius[0])[0]
        binary_threshold = np.quantile(self.mask, binary_quantile)
        binary_mask = (self.mask > binary_threshold)*255
        self.mask = binary_mask.astype(np.uint8)
        
    def rectangular_closing(self, min_angle=-70, max_angle=70, num_angles=15):
        angles = np.linspace(min_angle, max_angle, num_angles)
        for angle in angles:
            kernel = np.ones((20,2),np.uint8) 
            kernel = rotate(kernel, angle)
            self.mask = cv2.dilate(self.mask, kernel, iterations = 1)
            self.mask = cv2.erode(self.mask, kernel, iterations = 1)
            
    def skeleton(self):
        self.mask = (skeletonize(self.mask)*255).astype(np.uint8)    
        
    def crop_around_disc(self, diameter_multiplier: float = 4):
        try:
            # Convert (col, row) to Cartesian (x, y)
            discX = self.disc_x
            discY = self.disc_y
            
            if self.eye == "left":
                top_left_x = discX - (self.disc_diameter / 2)
                top_left_y = discY - (self.disc_diameter * diameter_multiplier)
                bottom_right_x = discX + (self.disc_diameter * diameter_multiplier)
                bottom_right_y = discY + (self.disc_diameter * diameter_multiplier)
            elif self.eye == "right":
                top_left_x = discX - (self.disc_diameter * diameter_multiplier)
                top_left_y = discY - (self.disc_diameter * diameter_multiplier)
                bottom_right_x = discX + (self.disc_diameter / 2)
                bottom_right_y = discY + (self.disc_diameter * diameter_multiplier)
                
            # Crop mask
            mask_pil = Image.fromarray(self.mask)
            self.mask = np.array(mask_pil.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y)))
            
        except Exception as e:
            print("Error cropping around disc:", str(e))
            print("Make sure that disc_x, disc_y and disc_diameter are specified in the database for this image.")
            
    def rotate(self, angle):
        """Rotate mask around its center by angle in degrees (positive values mean counter-clockwise rotation)."""
        mask_ui8 = self.mask.astype(np.uint8) * 255
        image_center = tuple(np.array(self.mask.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        self.mask = cv2.warpAffine(mask_ui8, rot_mat, self.mask.shape[1::-1], flags=cv2.INTER_LINEAR).astype(bool)
        
    ############ PRIVATE METHODS ############
    
    def _compute_areas(self):
        mask_ui8 = self.mask.astype(np.uint8) * 255
        output = cv2.connectedComponentsWithStats(mask_ui8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        areas = stats[:, cv2.CC_STAT_AREA][1:] # ignore the first element which corresponds to background pixels
        return areas

class ArcadeRANSAC(ArcadeDetector):
    
    def __init__(self, 
                 name: str, 
                 mask: np.ndarray, 
                 db_manager: DatabaseManager = None,
                 disc_center_x: float = None,
                 disc_center_y: float = None,
                 disc_diameter_px: float = None,
                 disc_fovea_angle_deg: float = None,
                 laterality: str = None,
                 seed: int = 42):
        np.random.seed(seed)
        super().__init__(
            name=name,
            mask=mask,
            db_manager=db_manager,
            disc_center_x=disc_center_x,
            disc_center_y=disc_center_y,
            disc_diameter_px=disc_diameter_px,
            disc_fovea_angle_deg=disc_fovea_angle_deg,
            laterality=laterality,
        )
    
    def __call__(self):
        
        # Mask filtering to isolate arcade vessels/shape
        super().__call__()
        
        # Get vessel coordinates
        coords = self.get_coordinates()
        self.x = coords[:,0]
        self.y = coords[:,1]
        self.x_reshaped = self.x.reshape((-1, 1))
        self.y_reshaped = self.y.reshape((-1, 1))
        
        # Model fitting
        self.fit_parabola()
        
    ############ PARABOLA ############
    
    def fit_parabola(self):
        self.quadratic = PolynomialFeatures(degree=2)
        self.y_quadratic_trans = self.quadratic.fit_transform(self.y_reshaped)
        self.parabola = RANSACRegressor(LinearRegression(fit_intercept=False), min_samples=15)
        self.parabola.fit(self.y_quadratic_trans, self.x_reshaped)
        # get coefficients
        self.c = self.parabola.estimator_.coef_[0, 0] 
        self.b = self.parabola.estimator_.coef_[0, 1] 
        self.concavity = self.parabola.estimator_.coef_[0, 2]


