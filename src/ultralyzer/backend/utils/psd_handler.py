import numpy as np
import photoshopapi as psapi
from photoshopapi import Layer_8bit
from definitions import PSD_LAYER_NAMES

################ HELPER METHODS ################

def extract_layers_from_psd(psd_path: str) -> dict[str, Layer_8bit]:
    # Read the PSD file
    file = psapi.LayeredFile_8bit.read(psd_path)

    # Extract layers into a dictionary
    layer_dict = {layer.name.lower(): layer for layer in file.layers}

    return layer_dict

def layer_to_mask(layer: Layer_8bit, 
                  alpha_threshold: int = 150) -> np.ndarray:
    # Extract image data from the layer
    layer_data = layer.get_image_data()
    
    # Create a mask from the RGBA data
    mask = np.array([layer_data[i] for i in range(3)]).transpose(1, 2, 0)
    mask = np.any(mask > 0, axis=-1).astype(np.uint8)
    alpha = layer_data[-1] > alpha_threshold
    mask[~alpha] = 0

    return mask.astype(np.uint8) * 255

def layer_to_image(layer: Layer_8bit) -> np.ndarray:
    layer_data = layer.get_image_data()
    return np.array([layer_data[i] for i in range(3)]).transpose(1, 2, 0)

################ EVENT METHODS ################

def process_psd_file(psd_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Process a PSD file to extract segmentation masks and color image.
    Both images are returned as numpy arrays in RGB format.
    """
    layers = extract_layers_from_psd(psd_path)
    
    processed_data = {}
    for name, layer in layers.items():
        if name.lower() in PSD_LAYER_NAMES[0:3]: # arteries, veins, optic disc
            processed_data[name] = layer_to_mask(layer)
        elif name.lower() == "color image":
            processed_data[name] = layer_to_image(layer)
            
    mask = np.stack([
        processed_data.get("arteries", np.zeros(processed_data["color image"].shape[:2], dtype=np.uint8)),
        processed_data.get("optic disc", np.zeros(processed_data["color image"].shape[:2], dtype=np.uint8)), 
        processed_data.get("veins", np.zeros(processed_data["color image"].shape[:2], dtype=np.uint8)),
    ], axis=-1)
    
    return mask, processed_data.get("color image", np.zeros_like(mask))
