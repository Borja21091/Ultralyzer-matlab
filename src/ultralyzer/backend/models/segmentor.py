import os
import numpy as np
from PIL import Image
from typing import Tuple, Any
import torch.nn.functional as F
from torchvision import tv_tensors
from abc import ABC, abstractmethod
from pathlib import PosixPath, PurePath
from torch.amp.autocast_mode import autocast
from backend.models.dinov3.model import AVSegmenter
from definitions import MODELS_DIR, MODEL_BASE_URL_UWF
from backend.utils.preprocessing import get_uwf_transform
from backend.utils.preprocessing import preprocess_uwf_disc_fov_loc_seg, process_uwf_disc_map, localise_centre_mass, process_uwf_fov_map

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import torch

from torchvision.transforms import v2 as T
from backend.models.models import SegmentationModel

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
    

class Segmentor(ABC):
    """Abstract base class for segmentation models"""
    
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
    
    @abstractmethod
    def segment(self, image: np.ndarray) -> Any:
        """
        Segment an image.
        
        Args:
            image: Input image (H, W, 3) as numpy array
        """
        pass


class UWFAVSegmentor(Segmentor):
    
    DEFAULT_MODEL_NAME = 'av_segmentation.pt'
    DEFAULT_MODEL_URL = MODEL_BASE_URL_UWF + '/' + DEFAULT_MODEL_NAME
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'uwf', DEFAULT_MODEL_NAME)
    

    def __init__(self, model_path=DEFAULT_MODEL_URL, threshold=DEFAULT_THRESHOLD, local_model_path=DEFAULT_MODEL_PATH):
        """
        Core inference class for UWF artery/vein segmentation model
        """
        super().__init__("uwf_av_segmentor", "1.0")
        self.CROP_LEFT = 450
        self.CROP_TOP = 650
        self.PATCH_H = 864
        self.PATCH_W = 992
        self.GAUSSIAN_SIGMA_H = 144
        self.GAUSSIAN_SIGMA_W = 165
        self.STRIDE_H = 672
        self.STRIDE_W = 800
        
        # R, G: ImageNet stats; R-G channel: centred at 0, std=0.5
        self._MEAN = (0.485, 0.456, 0.000)
        self._STD  = (0.229, 0.224, 0.500)
        
        self._patchsize = (self.PATCH_H, self.PATCH_W)
        self._batch = 32
        
        self._threshold = threshold
        self.device = DEVICE
        self.model = AVSegmenter().to(self.device)
        
        if not os.path.exists(local_model_path):
            torch.hub.load_state_dict_from_url(model_path, os.path.join(MODELS_DIR, 'uwf'), map_location=self.device)
        
        self.model.load_state_dict(torch.load(local_model_path, map_location=self.device))
            
        if self.device != "cpu":
            print("UWF Artery/Vein segmentation has been loaded with GPU acceleration!")
        self.model.eval()

    def __call__(self, x):
        """Direct call for inference on single image"""
        return self.segment(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
    
    ############ PROPERTIES ############
    
    @property
    def patchsize(self):
        return self._patchsize
    
    @patchsize.setter
    def patchsize(self, value: tuple[int, int]):
        self._patchsize = value
    
    @property
    def batch(self):
        return self._batch
    
    @batch.setter
    def batch(self, value: int):
        self._batch = value
    
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value
    
    ############ PUBLIC METHODS ############
    
    def segment(self, image) -> Tuple[np.ndarray, np.ndarray]:
        """Create segmentation mask for arteries and veins
        
        Returns:
        --------
            av_mask (np.ndarray): Colour-coded segmentation map (H, W, 3) with values in [0, 255]
            vessel_mask (np.ndarray): Binary vessel mask (H, W) where 1 indicates vessel pixels and 0 background
        """
        if isinstance(image, (str, PurePath, PosixPath)):
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
            
        H_orig, W_orig = image.shape[:2]
        
        # Border crop
        image_crop = self._border_crop(image, border_px=(self.CROP_LEFT, self.CROP_TOP))
        
        # Sliding window segmentation
        pred_crop = self._segment(image_crop)
        
        # Pad back to original size (border = background = 0)
        pred_full = np.zeros((H_orig, W_orig), dtype=np.int32)
        H_crop, W_crop = pred_crop.shape
        pred_full[self.CROP_TOP : self.CROP_TOP + H_crop, self.CROP_LEFT : self.CROP_LEFT + W_crop] = pred_crop
        
        # AV mask & vessel mask
        av_mask = np.zeros((H_orig, W_orig, 3), dtype=np.uint8)
        av_mask[pred_full == 1] = [255, 0, 0]   # Arteries in red
        av_mask[pred_full == 2] = [0, 0, 255]   # Veins in blue
        av_mask[pred_full == 3] = [255, 0, 255]   # Overlap in magenta (if needed)
        
        vessel_mask = (pred_full > 0).astype(np.uint8) * 255  # Binary mask where 255 indicates vessel pixels (arteries or veins)

        return av_mask, vessel_mask
    
    ############ PRIVATE METHODS ############
    
    def _border_crop(self, img: np.ndarray, border_px: Tuple[int, int] = (450, 650)) -> np.ndarray:
        """
        Crop the black scanning borders from an Optomap image.

        Args:
            img       : (H, W) or (H, W, C) array, any dtype.
            border_px : (horizontal_border, vertical_border) in pixels.
                        Removes h_border cols from left+right,
                                v_border rows from top+bottom.

        Returns:
            Cropped array with shape
            (H - 2*v_border, W - 2*h_border [, C]).
        """
        H, W = img.shape[:2]
        h_border, v_border = border_px

        if 2 * h_border >= W or 2 * v_border >= H:
            print(
                f"Warning: border_px={border_px} exceeds image dims ({W}×{H}). "
                "Returning original."
            )
            return img

        return img[v_border : H - v_border, h_border : W - h_border]

    def _make_gaussian_weight(self) -> np.ndarray:
        """
        2D Gaussian weight map for tile blending.

        Centre-weighted so that predictions near patch centres (fewest boundary
        artefacts) contribute more than edges.

        Returns:
            (h, w) float32 array.
        """
        sigma_h = self.GAUSSIAN_SIGMA_H
        sigma_w = self.GAUSSIAN_SIGMA_W
        h, w = self.patchsize
        cy, cx = h / 2.0, w / 2.0
        y = np.arange(h, dtype=np.float32) - cy
        x = np.arange(w, dtype=np.float32) - cx
        yy, xx = np.meshgrid(y, x, indexing="ij")
        g = np.exp(-0.5 * (yy ** 2 / sigma_h ** 2 + xx ** 2 / sigma_w ** 2))
        return g
    
    def _build_input_tensor(self, patch_rgb: np.ndarray) -> torch.Tensor:
        """
        Convert an RGB uint8 patch to a (1, 3, H, W) normalised (R, G, R−G) tensor.
        """
        f = patch_rgb.astype(np.float32) / 255.0
        R = f[:, :, 0]
        G = f[:, :, 1]
        RmG = R - G

        x = np.stack([R, G, RmG], axis=0)  # (3, H, W)
        for c in range(3):
            x[c] = (x[c] - self._MEAN[c]) / self._STD[c]

        return torch.from_numpy(x).unsqueeze(0).float()  # (1, 3, H, W)
    
    def _segment(self, 
                 image: np.ndarray,
                 num_classes: int = 4) -> np.ndarray:
        """
        Predict on a border-cropped image using overlapping tiles with Gaussian
        blending.

        Args:
            image : (H_crop, W_crop, 3) uint8 RGB, already border-cropped.
            num_classes: number of output classes.

        Returns:
            pred_classes : (H_crop, W_crop) int32 with values in {0,1,2,3}.
        """
        h, w = image.shape[:2]
        gauss = self._make_gaussian_weight()
        
        # Accumulation canvases
        prob_canvas = np.zeros((num_classes, h, w), dtype=np.float64)
        weight_canvas = np.zeros((h, w), dtype=np.float64)
        
        # Generate tiple positions
        rows = list(range(0, h - self.PATCH_H + 1, self.STRIDE_H))
        if rows[-1] + self.PATCH_H < h:
            rows.append(h - self.PATCH_H)

        cols = list(range(0, w - self.PATCH_W + 1, self.STRIDE_W))
        if cols[-1] + self.PATCH_W < w:
            cols.append(w - self.PATCH_W)
            
        for top in rows:
            for left in cols:
                # Extract patch (zero-pad not needed since we clip to image)
                patch = image[top : top + self.PATCH_H, left : left + self.PATCH_W]

                # Pad if patch is smaller than expected (edge case)
                ph, pw = patch.shape[:2]
                if ph < self.PATCH_H or pw < self.PATCH_W:
                    padded = np.zeros((self.PATCH_H, self.PATCH_W, 3), dtype=np.uint8)
                    padded[:ph, :pw] = patch
                    patch = padded

                # Forward pass
                x = self._build_input_tensor(patch).to(self.device)
                with torch.no_grad(), autocast("cuda", dtype=torch.float16):
                    logits = self.model({"image": x})  # (1, 4, PATCH_H, PATCH_W)

                probs = F.softmax(logits.float(), dim=1).cpu().numpy()[0]  # (4, PH, PW)

                # Accumulate with Gaussian weighting
                g = gauss[:ph, :pw]  # clip gauss if patch was clipped
                for c in range(num_classes):
                    prob_canvas[c, top : top + ph, left : left + pw] += probs[c, :ph, :pw] * g
                weight_canvas[top : top + ph, left : left + pw] += g

        # Normalise and argmax
        weight_canvas = np.maximum(weight_canvas, 1e-8)
        for c in range(num_classes):
            prob_canvas[c] /= weight_canvas

        pred_classes = prob_canvas.argmax(axis=0).astype(np.int32)
        return pred_classes


class UWFDiscLocaliser(Segmentor):
    
    DEFAULT_MODEL_NAME = 'od_localisation.pt'
    DEFAULT_MODEL_URL = MODEL_BASE_URL_UWF + '/' + DEFAULT_MODEL_NAME
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'uwf', DEFAULT_MODEL_NAME)
    
    def __init__(self, model_path=DEFAULT_MODEL_URL, threshold=DEFAULT_THRESHOLD, local_model_path=DEFAULT_MODEL_PATH):
        """
        Core inference class for UWF rough disc localisation.
        """
        super().__init__("uwf_disc_localiser", "1.0")
        self._patchsize = 512
        self._batch = 32
        self.transform = get_uwf_transform(size=(512, 512))
        self._threshold = threshold
        self.device = DEVICE
        self.model = SegmentationModel('segformer', 'resnet34', in_channels=1).to(self.device)
        
        if not os.path.exists(local_model_path):
            torch.hub.load_state_dict_from_url(model_path, os.path.join(MODELS_DIR, 'uwf'), map_location=self.device)
        
        self.model.load_state_dict(torch.load(local_model_path, map_location=self.device))
            
        if self.device != "cpu":
            print("UWF disc localisation has been loaded with GPU acceleration!")
        self.model.eval()
        
    def __call__(self, x):
        """Direct call for inference on single image"""
        return self.segment(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
    
    ############ PROPERTIES ############
    
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value
    
    ############ PUBLIC METHODS ############
    
    def segment(self, img, soft_pred=False) -> Tuple:
        """Segment disc in the image.
        
        Returns:
        --------
            pred (np.ndarray): Segmentation mask
            loc (tuple): (y, x) coordinates of the disc centre in the original image"""
        if isinstance(img, (str, PurePath, PosixPath)):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert('RGB')
        elif isinstance(img, Image.Image):
            img = img.convert('RGB')
            
        # Preprocess image
        img, tl = preprocess_uwf_disc_fov_loc_seg(img)
        img_shape = (img.height, img.width)
        
        # If downsamples to (1024, 1024), prepare for upsampling
        RESIZE = T.Resize(img_shape, antialias=True)
        
        with torch.no_grad():
            img, (M, N) = self.transform(img)
            img = img.unsqueeze(0).to(self.device)
            pred = self.model(img).squeeze(0).sigmoid()[:, :M, :N]
            
            # Resize back to native resolution
            pred = RESIZE(tv_tensors.Image(pred))[0]
                
            # Return if soft_pred, otherwise post-process
            if soft_pred:
                return (pred.cpu().numpy(), None)
            else:
                pred = (pred > self.threshold).squeeze().cpu().numpy().astype(np.uint8)
                pred = process_uwf_disc_map(pred)
                loc = localise_centre_mass(pred) # Location in cropped image
                loc = (loc[0] + tl[0], loc[1] + tl[1]) # (row, col) -> (y, x) # Location in original image
                return (pred, loc)


class UWFDiscDetailedSegmenter(Segmentor):
    
    DEFAULT_MODEL_NAME = 'od_segmentation.pt'
    DEFAULT_MODEL_URL = MODEL_BASE_URL_UWF + '/' + DEFAULT_MODEL_NAME
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'uwf', DEFAULT_MODEL_NAME)
    
    def __init__(self, model_path=DEFAULT_MODEL_URL, threshold=DEFAULT_THRESHOLD, local_model_path=DEFAULT_MODEL_PATH):
        """
        Core inference class for UWF detailed disc segmentation.
        """
        super().__init__("uwf_disc_seg", "1.0")
        self.transform = get_uwf_transform(size=(256, 256))
        self._threshold = threshold
        self.device = DEVICE
        self.model = SegmentationModel('segformer', 'resnet34', in_channels=1).to(self.device)
        
        if not os.path.exists(local_model_path):
            torch.hub.load_state_dict_from_url(model_path, os.path.join(MODELS_DIR, 'uwf'), map_location=self.device)
        
        self.model.load_state_dict(torch.load(local_model_path, map_location=self.device))
            
        if self.device != "cpu":
            print("UWF disc segmentation has been loaded with GPU acceleration!")
        self.model.eval()
        
    def __call__(self, x: Tuple):
        """Direct call for inference on single image"""
        return self.segment(*x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
    
    ############ PROPERTIES ############
    
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value
    
    ############ PUBLIC METHODS ############
    
    @torch.inference_mode()
    def segment(self, img, od_centre: tuple[float], soft_pred=False):
        """
        Inference on a single image
        """
        if isinstance(img, (str, PurePath, PosixPath)):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert('RGB')
        elif isinstance(img, Image.Image):
            img = img.convert('RGB')
            
        # Preprocess image
        img, _ = preprocess_uwf_disc_fov_loc_seg(img, centre=od_centre, crop_size=(256, 256))
        img_shape = (img.height, img.width)
        
        # If downsamples to (256, 256), prepare for upsampling
        if img_shape != (256, 256):
            RESIZE = T.Resize(img_shape, antialias=True)
        else:
            RESIZE = None
        
        with torch.no_grad():
            img, (M, N) = self.transform(img)
            img = img.unsqueeze(0).to(self.device)
            pred = self.model(img).squeeze(0).sigmoid()[:, :M, :N]
            
            # Resize back to native resolution
            if RESIZE is not None:
                pred = RESIZE(tv_tensors.Image(pred))[0]
                
            # Return if soft_pred, otherwise post-process
            if soft_pred:
                return pred.cpu().numpy()
            else:
                pred = (pred > self.threshold).squeeze().cpu().numpy().astype(np.uint8)
                pred = process_uwf_disc_map(pred)
                return pred
    

class UWFDiscSegmentor(Segmentor):
    """Wrapper class for UWF disc segmentation combining localisation and detailed segmentation"""
    
    def __init__(self, 
                 localiser: UWFDiscLocaliser = None, 
                 segmenter: UWFDiscDetailedSegmenter = None):
        """
        Core inference class for UWF disc segmentation.
        """
        super().__init__("uwf_disc_full_seg", "1.0")
        self.localiser = localiser if localiser is not None else UWFDiscLocaliser()
        self.segmenter = segmenter if segmenter is not None else UWFDiscDetailedSegmenter()
    
    def __call__(self, x):
        """Direct call for inference on single image"""
        return self.segment(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
    ############ PUBLIC METHODS ############
    
    def segment(self, image) -> np.ndarray:
        """Segment disc in the image."""
        if isinstance(image, (str, PurePath, PosixPath)):
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
            
        # First localise disc
        _, loc = self.localiser.segment(image)
        
        # Then segment disc in detail
        disc_mask = self.segmenter.segment(image, od_centre=loc)
        h, w = disc_mask.shape[:2]
        
        # Prepare output
        output = np.zeros_like(image[:, :, 0], dtype=np.uint8)
        
        # Place disc mask in output
        output[int(loc[0]) - h//2 : int(loc[0]) + h//2, 
               int(loc[1]) - w//2 : int(loc[1]) + w//2] = disc_mask
        
        return output


class UWFFoveaLocaliser(Segmentor):
    
    DEFAULT_MODEL_NAME = 'fovea_localisation.pt'
    DEFAULT_MODEL_URL = MODEL_BASE_URL_UWF + '/' + DEFAULT_MODEL_NAME
    DEFAULT_THRESHOLD = 0.5
    DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'uwf', DEFAULT_MODEL_NAME)
    
    def __init__(self, model_path=DEFAULT_MODEL_URL, threshold=DEFAULT_THRESHOLD, local_model_path=DEFAULT_MODEL_PATH):
        """
        Core inference class for UWF rough disc localisation.
        """
        super().__init__("uwf_fovea_localiser", "1.0")
        self.transform = get_uwf_transform(size=(512, 512))
        self._threshold = threshold
        self.device = DEVICE
        self.model = SegmentationModel('segformer', 'resnet34', in_channels=1).to(self.device)
        
        if not os.path.exists(local_model_path):
            torch.hub.load_state_dict_from_url(model_path, os.path.join(MODELS_DIR, 'uwf'), map_location=self.device)
        
        self.model.load_state_dict(torch.load(local_model_path, map_location=self.device))
            
        if self.device != "cpu":
            print("UWF fovea localisation has been loaded with GPU acceleration!")
        self.model.eval()
        
    def __call__(self, x):
        """Direct call for inference on single image"""
        return self.segment(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
    
    ############ PROPERTIES ############
    
    @property
    def threshold(self):
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: float):
        self._threshold = value
    
    ############ PUBLIC METHODS ############
    
    @torch.inference_mode()
    def segment(self, img, soft_pred=False) -> Tuple:
        """
        Inference on a single image
        """
        if isinstance(img, (str, PurePath, PosixPath)):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert('RGB')
        elif isinstance(img, Image.Image):
            img = img.convert('RGB')
        
        # Preprocess image
        img, tl = preprocess_uwf_disc_fov_loc_seg(img)
        img_shape = (img.height, img.width)
        
        # If downsamples to (1024, 1024), prepare for upsampling
        RESIZE = T.Resize(img_shape, antialias=True)
        
        with torch.no_grad():
            img, (M, N) = self.transform(img)
            img = img.unsqueeze(0).to(self.device)
            pred = self.model(img).squeeze(0).sigmoid()[:, :M, :N]
            
            # Resize back to native resolution
            pred = RESIZE(tv_tensors.Image(pred))[0]
                
            # Return if soft_pred, otherwise post-process
            if soft_pred:
                return (pred.cpu().numpy(), None, None)
            else:
                pred = (pred > self.threshold).squeeze().cpu().numpy().astype(np.uint8)
                pred = process_uwf_fov_map(pred)
                loc = localise_centre_mass(pred) # Location in cropped image
                loc = (loc[0] + tl[0], loc[1] + tl[1]) # (row, col) -> (y, x) # Location in original image
                return (pred, loc, tl)


class UWFFoveaSegmentor(Segmentor):
    """Wrapper class for UWF fovea segmentation"""
    
    def __init__(self, 
                 localiser: UWFFoveaLocaliser = None):
        """
        Core inference class for UWF fovea segmentation.
        """
        super().__init__("uwf_fovea_full_seg", "1.0")
        self.localiser = localiser if localiser is not None else UWFFoveaLocaliser()
    
    def __call__(self, x):
        """Direct call for inference on single image"""
        return self.segment(x)
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
    ############ PUBLIC METHODS ############
    
    def segment(self, image) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Segment fovea in the image."""
        if isinstance(image, (str, PurePath, PosixPath)):
            image = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
            
        # First localise fovea
        pred, loc, tl = self.localiser.segment(image)
        h, w = pred.shape[:2]
        
        # Prepare output
        output = np.zeros_like(image[:, :, 0], dtype=np.uint8)
        
        # Place fovea location in output
        output[tl[0]:tl[0] + h, tl[1]:tl[1] + w] = pred
        
        return (output, loc)
