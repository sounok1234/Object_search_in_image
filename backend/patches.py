import os
import jsonlines
import tempfile
import cv2
import numpy as np

def split_image_into_patches(np_image, patch_height, patch_width, overlap=0.5, init = 0):
    # Read the image
    image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    data = []
    temp_dir = tempfile.mkdtemp()

    step_y = int(patch_height * (1 - overlap))
    step_x = int(patch_width * (1 - overlap))

    for y in range(0, height - patch_height + 1, step_y):
        for x in range(0, width - patch_width + 1, step_x):
            patch = image[y:y + patch_height, x:x + patch_width]
            gray_img = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            if np.any(gray_img == 0):
                patch_filename = f"patch_{init}.jpg"
                patch_path = os.path.join(temp_dir, patch_filename)

                resized_patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(patch_path, resized_patch)

                patch_json = {
                    "file_name": patch_filename,
                    "file_path": patch_path,
                    "coords": [(x, y), (x + patch_width, y + patch_height)]
                }
                data.append(patch_json)
                init += 1

    json_file_path = os.path.join(temp_dir, 'metadata.jsonl')
    with jsonlines.open(json_file_path, mode='w') as writer:
        writer.write_all(data)

    return temp_dir
