import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as TF


class FixShape(T.Transform):
    def __init__(self, factor=32):
        """Forces input to have dimensions divisble by 32"""
        super().__init__()
        self.factor = factor

    def __call__(self, img):
        M, N = img.shape[-2:]
        pad_M = (self.factor - M%self.factor) % self.factor
        pad_N = (self.factor - N%self.factor) % self.factor
        return TF.pad(img, padding=(0, 0, pad_N, pad_M)), (M, N)

    def __repr__(self):
        return self.__class__.__name__


def rough_crop(img: Image.Image, crop_size: tuple, centre: tuple = None) -> tuple:
    """
    Crop the image to the specified size.
    """
    width, height = img.size
    if centre is None:
        centre = (height // 2, width // 2) # (row, col)
        
    # Crop margins
    left = centre[1] - crop_size[1] // 2
    top = centre[0] - crop_size[0] // 2
    right = centre[1] + crop_size[1] // 2
    bottom = centre[0] + crop_size[0] // 2
    
    return img.crop((left, top, right, bottom)), (top, left)

def localise_centre_mass(map) -> tuple:
    """
    Find the centre of mass of the largest connected component in a binary map.
    
    Args:
        map (np.ndarray): Binary image.
        
    Returns:
        tuple: (y, x)(row, col) coordinates of the centre of mass.
    """
    # Find contours
    contours, _ = cv2.findContours(map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate moments
    M = cv2.moments(largest_contour)
    
    if M['m00'] == 0:
        return None
    
    # Calculate centre of mass
    cX = M['m10'] / M['m00']
    cY = M['m01'] / M['m00']
    
    return (cY, cX)

# ------------------------- UWF ------------------------- #

def get_uwf_transform(size=(256, 256)):
    """Tensor, dimension and normalisation default augs for UWF Vessel Segmentation"""
    return T.Compose([
        T.PILToTensor(),
        T.Resize(size=size, antialias=True),
        T.ToDtype(torch.float32, scale=True),
        FixShape(factor=32),
        T.Normalize(mean=(0.5,), std=(0.5,))
    ])

################### DISC ###################

def preprocess_uwf_disc_fov_loc_seg(img: Image.Image, crop_size=(1024, 1024), centre=None) -> tuple:
    """
    Preprocess the UWF disc image for localisation.
    """
    # Split channels
    _, g, _ = img.split()

    # CLAHE for green channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_clahe = Image.fromarray(clahe.apply(np.array(g))).convert('L')
    
    # Rough crop
    g_clahe_crop, tl = rough_crop(g_clahe, crop_size, centre)
    
    return g_clahe_crop, tl
    
def process_uwf_disc_map(map):
    """
    Post-process the UWF disc map.
    """
    # Keep largest connected component
    num_labels, labels_im = cv2.connectedComponents(map.astype(np.uint8), connectivity=8)
    largest_label = 1 + np.argmax(np.bincount(labels_im.flat)[1:])
    map = np.zeros_like(map)
    map[labels_im == largest_label] = 1
    
    # Fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    map = cv2.morphologyEx(map, cv2.MORPH_CLOSE, kernel)
    
    # Edge smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    map = cv2.morphologyEx(map, cv2.MORPH_OPEN, kernel)
    map = cv2.GaussianBlur(map, (5, 5), 0)
    _, map = cv2.threshold(map, 0.5, 1, cv2.THRESH_BINARY)
    map = map.astype(np.uint8) * 255
    
    return map

################### FOVEA ###################

def process_uwf_fov_map(map):
    """
    Post-process the UWF fovea map.
    """
    # Keep largest connected component
    num_labels, labels_im = cv2.connectedComponents(map.astype(np.uint8), connectivity=8)
    largest_label = 1 + np.argmax(np.bincount(labels_im.flat)[1:])
    map = np.zeros_like(map)
    map[labels_im == largest_label] = 1
    
    # Fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    map = cv2.morphologyEx(map, cv2.MORPH_CLOSE, kernel)
    
    # Edge smoothing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    map = cv2.morphologyEx(map, cv2.MORPH_OPEN, kernel)
    map = cv2.GaussianBlur(map, (5, 5), 0)
    _, map = cv2.threshold(map, 0.5, 1, cv2.THRESH_BINARY)
    map = map.astype(np.uint8) * 255
    
    return map