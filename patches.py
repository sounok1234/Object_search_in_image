import tempfile
import cv2
import numpy as np

def split_image_into_patches(np_image, patch_size=48, overlap=0.5, init = 0):
    # Read the image
    image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    data = []
    temp_dir = tempfile.mkdtemp()

    # Calculate overlap size in pixels
    overlap_pixels = int(patch_size * overlap)
    # Generate patches with overlap
    for y in range(0, height-patch_size+1, patch_size - overlap_pixels):
        for x in range(0, width-patch_size+1, patch_size - overlap_pixels):
            patch = image[y:y+patch_size, x:x+patch_size]  
            gray_img = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if np.any(gray_img == 0):     
                patch_filename = f"patch_{init}.jpg"
                patch_path = f"{temp_dir}/{patch_filename}"

                # Resize patch and save temporarily
                resized_patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(patch_path, resized_patch)

                # Store patch metadata
                patch_json = {
                    "file_name": patch_filename,
                    "file_path": patch_path,
                    "coords": [(x, y), (x + patch_size, y + patch_size)]
                }
                data.append(patch_json)
                init += 1

    return temp_dir, data
