import json
import shutil
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoModel, ViTImageProcessor
import cv2
import numpy as np
import os
from utils import DatasetWithEmbeddings
from patches import split_image_into_patches
from datasets import load_dataset
import gc

# Fix OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

app = FastAPI()
# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change this for security
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"--> {request.method} {request.url}")
    print("Headers:", dict(request.headers))

    # Read body for logging (must be done carefully)
    if request.method == "POST":
        try:
            body = await request.body()
            print(f"Request body (raw): {body[:200]}...")
        except Exception as e:
            print("Could not read body:", e)

    response = await call_next(request)
    return response

@app.post("/find_objects/")
async def find_objects(
    uuid: str = Form(...), 
    file: UploadFile = File(...), 
    rectangle: str = Form(...)
):
    """
    User selects a bounding box on the image to search for similar objects.
    """
    # Load the uploaded image
    image = file.file.read()
    nparr = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="No image uploaded")    

    rect_data = json.loads(rectangle)
    x1, y1, x2, y2 = rect_data["x1"], rect_data["y1"], rect_data["x2"], rect_data["y2"]

    # Initialize model and image processor
    extractor = ViTImageProcessor.from_pretrained("./models/deit-small") 
    model = AutoModel.from_pretrained("./models/deit-small")
    model.config.hidden_size

    # Extract selected patch
    selected_patch = image[y1:y2, x1:x2]
    # max_dim = max(selected_patch.shape)
    patch_height, patch_width = selected_patch.shape[:2]
    query_patch = cv2.resize(selected_patch, (224, 224))
    # Call function to split image into patches
    temp_dir = split_image_into_patches(image, patch_height, patch_width)
    # Get similar patches
    dataset = load_dataset("imagefolder", data_dir=temp_dir)
    train_dataset = dataset['train']
    dataset_init = DatasetWithEmbeddings(extractor, model)
    dataset_with_embeddings = train_dataset.map(lambda example: 
                            {'embeddings': dataset_init.extract_embeddings(example['image'])})
    dataset_with_embeddings.add_faiss_index(column='embeddings')
    boxes = dataset_init.get_neighbors(dataset_with_embeddings, query_patch, 50)

    # Clear memory and local database
    del dataset_with_embeddings, dataset_init, train_dataset, dataset
    gc.collect()
    shutil.rmtree(temp_dir, ignore_errors=True)
    # Send data to frontend
    return JSONResponse(content={
        "message": "File received",
        "uuid": uuid,
        "boxes": boxes,
    })
