import numpy as np
import pydicom as dcm

################ HELPER METHODS ################



################ EVENT METHODS ################

def dicom_to_image(dicom_path: str) -> tuple[bool, np.ndarray, dict]:
    data = dcm.dcmread(dicom_path)
    
    # Check if pixel data exists
    if not hasattr(data, 'pixel_array') or data.pixel_array is None:
        return False, np.array([]), {}
    
    # Parse metadata
    metadata = {
        'PatientID': data.get('PatientID', 'Unknown'),
        'Laterality': data.get('ImageLaterality', 'Unknown'),
        'Timestamp': data.get('AcquisitionDateTime', 'Unknown')
    }
    
    # Extract pixel data
    image = data.pixel_array
    
    return True, image, metadata