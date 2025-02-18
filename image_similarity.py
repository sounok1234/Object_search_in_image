from io import BytesIO
import os
import tempfile
from datasets import load_dataset, Dataset
import faiss
from google.cloud import storage
import numpy as np
import pyarrow.parquet as pq

# output_dir = "patches_train"
# model_ckpt = "facebook/deit-small-patch16-224"

# extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
# model = AutoModel.from_pretrained(model_ckpt)
# hidden_dim = model.config.hidden_size
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "object-search-image.json"
GCS_BUCKET_NAME = "faiss-storage-1234"
GCS_EMBEDDINGS_FILE = "embeddings/old_embeddings"
GCS_FAISS_INDEX_FILE = "embeddings/old_index.faiss"

storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

def upload_to_gcs(destination_blob_name, data):
    blob = bucket.blob(destination_blob_name)
    data.seek(0)  # Ensure buffer is at the start
    blob.upload_from_file(data, rewind=True)
    print(f"File uploaded to {destination_blob_name}")    

def download_from_gcs(blob_name):
    """Downloads a file from Google Cloud Storage into memory."""
    blob = bucket.blob(blob_name)
    data = BytesIO()
    blob.download_to_file(data)
    data.seek(0)  # Reset pointer to the beginning
    print(f"Downloaded {blob_name} into memory.")
    return data

def load_embeddings_from_parquet(parquet_data):
    """Load embeddings from an in-memory Parquet file."""
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_data)
    embeddings = table["embeddings"].to_numpy()  # Extract embeddings column as a NumPy array
    return embeddings

def extract_embeddings(image, extractor, model):
    image_pp = extractor(image, return_tensors="pt")
    features = (model(**image_pp).last_hidden_state[:, 0].detach().numpy()).squeeze()
    return features.squeeze()

def create_embeddings(output_dir, extractor, model):    
    dataset = load_dataset("imagefolder", data_dir=output_dir)
    train_dataset = dataset['train']    
    dataset_with_embeddings = train_dataset.map(lambda example: 
                            {'embeddings': extract_embeddings(example["image"], extractor, model)})
    # Convert dataset to an in-memory Parquet file
    embeddings_data = BytesIO()
    table = dataset_with_embeddings.data.table
    pq.write_table(table, embeddings_data)
    # Upload dataset embeddings to GCS
    upload_to_gcs(GCS_EMBEDDINGS_FILE, embeddings_data)
    # Extract embeddings into a NumPy array
    embeddings_list = [np.array(e, dtype=np.float32) for e in dataset_with_embeddings["embeddings"]]
    embeddings_array = np.stack(embeddings_list)
    # Create FAISS index
    dimension = embeddings_array.shape[1]  
    index = faiss.IndexFlatL2(dimension) 
    index.add(embeddings_array)  
    # Save FAISS index to memory
    faiss_buffer = BytesIO(faiss.serialize_index(index))
    # Upload FAISS index to GCS from memory
    upload_to_gcs(GCS_FAISS_INDEX_FILE, faiss_buffer)

    return {"message": "Embeddings and FAISS index saved & uploaded to GCS!"}

def get_embeddings_and_faiss_index():
    embeddings_data = download_from_gcs(GCS_EMBEDDINGS_FILE)
    embeddings_array = load_embeddings_from_parquet(embeddings_data)
    faiss_data = download_from_gcs(GCS_FAISS_INDEX_FILE)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(faiss_data.getvalue())  # Write the binary data to the temporary file
        tmp_file.flush()  # Ensure the data is written to disk
        tmp_file_path = tmp_file.name
    # Load the FAISS index from the temporary file
    faiss_index = faiss.read_index(tmp_file_path)
    print(faiss_index.ntotal, "vectors loaded from FAISS index.")
    # Delete the temporary file manually
    os.remove(tmp_file_path)
    # Create a dataset (if you don't already have one)
    dataset = Dataset.from_dict({"embeddings": embeddings_array})
    # Add the FAISS index to the dataset
    print(len(dataset), "vectors loaded from dataset.")
    dataset.add_faiss_index("embeddings", custom_index=faiss_index)
    
    return dataset

def get_neighbors(extractor, model, dataset_with_embeddings, query_image, top_k):
    query_embedding = model(**extractor(query_image, return_tensors="pt"))
    query_embedding = query_embedding.last_hidden_state[:, 0].detach().numpy().squeeze()
    print(query_embedding.shape)
    print(dataset_with_embeddings.shape)
    scores, retrieved_examples = dataset_with_embeddings.get_nearest_examples('embeddings', query_embedding, k=top_k)
    return scores, retrieved_examples

