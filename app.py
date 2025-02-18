from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from transformers import AutoFeatureExtractor, AutoModel
# from typing import List
import cv2
import numpy as np
import os
# from PIL import Image
from image_similarity import get_embeddings_and_faiss_index, get_neighbors, create_embeddings
from patches import split_image_into_patches
from datasets import Dataset

# Fix OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "object-search-image.json"

GCS_EMBEDDINGS_FILE = "embeddings/old_embeddings"
GCS_FAISS_INDEX_FILE = "embeddings/old_index.faiss"

app = FastAPI()

# Store image and embeddings 
app.state.image = None
model_ckpt = "facebook/deit-small-patch16-224"
app.state.extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
app.state.model = AutoModel.from_pretrained(model_ckpt)
app.state.hidden_dim = app.state.model.config.hidden_size
# app.state.dataset_with_embeddings = None  # Properly store dataset

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """
    Uploads an image, splits it into patches, and generates embeddings.
    """
    # Read the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    app.state.image = image
    
    # Call function to split image into patches
    # temp_dir, patches_info = split_image_into_patches(image)
    # extractor = app.state.extractor
    # model = app.state.model
    # create_embeddings(temp_dir, extractor, model)

    return {
        "message": "Image uploaded successfully!",
        # "patches_directory": temp_dir,
        # "patches_info": patches_info
    }    

@app.post("/select/")
async def select_patch(x1: int = Form(...), y1: int = Form(...), x2: int = Form(...), y2: int = Form(...)):
    """
    User selects a bounding box on the image to search for similar objects.
    """
    """Load dataset embeddings & FAISS index from GCS."""    

    # Load the uploaded image
    image = app.state.image
    if image is None:
        raise HTTPException(status_code=400, detail="No image uploaded")

    # Extract selected patch
    selected_patch = image[y1:y2, x1:x2]
    max_dim = max(selected_patch.shape)
    query_patch = cv2.resize(selected_patch, (224, 224))

    # Call function to split image into patches
    temp_dir, patches_info = split_image_into_patches(image, max_dim)
    extractor = app.state.extractor
    model = app.state.model
    create_embeddings(temp_dir, extractor, model)
    dataset_embeddings = get_embeddings_and_faiss_index()
    # Find similar objects
    scores, retrieved_examples = get_neighbors(extractor, model, dataset_embeddings, query_patch, 3)
    coords = np.array(retrieved_examples['coords'])
    # Normalize scores
    min_val, max_val = np.min(scores), np.max(scores)
    normalized_scores = [(1 - (s - min_val) / (max_val - min_val)) for s in scores]

    # Get bounding boxes
    boxes = [[c[0][0], c[0][1], c[1][0], c[1][1]] for c in coords.tolist()]
    
    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, normalized_scores, score_threshold=0.000001, nms_threshold=0.49)
    
    # Draw rectangles for detected objects
    for i in indices:
        x, y, x2, y2 = boxes[i]
        cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 2)

    # Save output
    cv2.imwrite("output.jpg", image)

    return {"message": "Similar objects found", "output_image": "output.jpg"}

@app.get("/download/")
async def download_output():
    """
    Endpoint to download the json for the bboxes for the original image
    """
    if not os.path.exists("output.jpg"):
        raise HTTPException(status_code=400, detail="No processed image available.")
    return {"output_image": "output.jpg"}