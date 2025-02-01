import cv2
import os
import jsonlines
import numpy as np

def split_image_into_patches(np_image, output_dir, patch_size=48, overlap=0.5, init = 0):
    # Read the image
    image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    data = []

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Calculate overlap size in pixels
    overlap_pixels = int(patch_size * overlap)
    # Generate patches with overlap
    for y in range(0, height-patch_size+1, patch_size - overlap_pixels):
        for x in range(0, width-patch_size+1, patch_size - overlap_pixels):
            patch = image[y:y+patch_size, x:x+patch_size]  
            gray_img = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if np.any(gray_img == 0):     
                patch_json = {"file_name": "patch_" + str(init) + ".jpg",  
                            "coords" : [(x, y), (x + patch_size, y + patch_size)]}
                data.append(patch_json)
                patch_filename = os.path.join(output_dir, "patch_" + str(init) + ".jpg")
                resized_patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(patch_filename, resized_patch)

                init += 1

    json_file_path = os.path.join(output_dir, 'metadata.jsonl')
    with jsonlines.open(json_file_path, mode='w') as writer:
        writer.write_all(data)

img = cv2.imread('1_left.jpg')
split_image_into_patches(img, "patches_train")