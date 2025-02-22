from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

# Initialize image 
app.state.image = None
app.state.boxes = None

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

    return {"message": "Image uploaded successfully!"}    

@app.post("/select/")
async def select_patch(x1: int = Form(...), y1: int = Form(...), x2: int = Form(...), 
                       y2: int = Form(...), num_of_objects: int = Form(...)):
    """
    User selects a bounding box on the image to search for similar objects.
    """
    """Load dataset embeddings & FAISS index from GCS."""    

    # Load the uploaded image
    image = app.state.image
    if image is None:
        raise HTTPException(status_code=400, detail="No image uploaded")
    
    model_ckpt = "facebook/deit-small-patch16-224"
    extractor = ViTImageProcessor.from_pretrained(model_ckpt) 
    model = AutoModel.from_pretrained(model_ckpt)
    model.config.hidden_size

    # Extract selected patch
    selected_patch = image[y1:y2, x1:x2]
    max_dim = max(selected_patch.shape)
    query_patch = cv2.resize(selected_patch, (224, 224))
    # Call function to split image into patches
    temp_dir = split_image_into_patches(image, max_dim)
    # Get similar patches
    dataset = load_dataset("imagefolder", data_dir=temp_dir)
    train_dataset = dataset['train']
    dataset_init = DatasetWithEmbeddings(extractor, model)
    dataset_with_embeddings = train_dataset.map(lambda example: 
                            {'embeddings': dataset_init.extract_embeddings(example['image'])})
    dataset_with_embeddings.add_faiss_index(column='embeddings')
    boxes, indices = dataset_init.get_neighbors(dataset_with_embeddings, query_patch, num_of_objects)
    app.state.boxes = boxes
    # Clear memory
    del dataset_with_embeddings, dataset_init, train_dataset, dataset
    gc.collect()
    # Draw rectangles for detected objects
    for i in indices:
        x, y, x2, y2 = boxes[i]
        cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 2)

    return {"message": "Similar objects found"}

@app.get("/download_json/")
async def download_output():
    """
    Endpoint to download the json for the bboxes for the original image
    """
    if not os.path.exists("output.jpg"):
        raise HTTPException(status_code=400, detail="No processed image available.")
    return {"output_image": "output.jpg"}
