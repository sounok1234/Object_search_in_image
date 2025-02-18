from io import BytesIO
import os
import tempfile
import faiss
from google.cloud import storage
from datasets import load_dataset
import numpy as np
from datasets import Dataset
from transformers import AutoFeatureExtractor, AutoModel
import pyarrow.parquet as pq

from image_similarity import extract_embeddings

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "object-search-image.json"
model_ckpt = "facebook/deit-small-patch16-224"
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
model.config.hidden_size

GCS_BUCKET_NAME = "faiss-storage-1234"
GCS_EMBEDDINGS_FILE = "embeddings/old_embeddings"
GCS_FAISS_INDEX_FILE = "embeddings/old_index.faiss"

storage_client = storage.Client()
bucket = storage_client.bucket(GCS_BUCKET_NAME)

# class BytesIOReader(faiss.VectorIOReader):
#     def __init__(self, data):
#         super().__init__()
#         self.data = data

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         pass

#     def read(self, size):
#         return self.data.read(size)

def upload_to_gcs_io(destination_blob_name, data):
    blob = bucket.blob(destination_blob_name)
    data.seek(0)  # Ensure buffer is at the start
    blob.upload_from_file(data, rewind=True)
    print(f"File uploaded to {destination_blob_name}")
    

# def download_from_gcs(blob_name, destination_file_name):
#     """Downloads a file from Google Cloud Storage."""
#     blob = bucket.blob(blob_name)
#     blob.download_to_filename(destination_file_name)
#     print(f"Downloaded {blob_name} to {destination_file_name}")

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

def create_embeddings_io(output_dir, extractor, model):    
    dataset = load_dataset("imagefolder", data_dir=output_dir)
    train_dataset = dataset['train']    
    dataset_with_embeddings = train_dataset.map(lambda example: 
                            {'embeddings': extract_embeddings(example["image"], extractor, model)})
    # Convert dataset to an in-memory Parquet file
    embeddings_data = BytesIO()
    table = dataset_with_embeddings.data.table
    pq.write_table(table, embeddings_data)
    # Upload dataset embeddings to GCS
    upload_to_gcs_io(GCS_EMBEDDINGS_FILE, embeddings_data)
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
    upload_to_gcs_io(GCS_FAISS_INDEX_FILE, faiss_buffer)

    return {"message": "Embeddings and FAISS index saved & uploaded to GCS!"}

embeddings_data = download_from_gcs(GCS_EMBEDDINGS_FILE)
embeddings_array = load_embeddings_from_parquet(embeddings_data)
faiss_data = download_from_gcs(GCS_FAISS_INDEX_FILE)
with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    tmp_file.write(faiss_data.getvalue())  # Write the binary data to the temporary file
    tmp_file.flush()  # Ensure the data is written to disk
    tmp_file_path = tmp_file.name

# Load the FAISS index from the temporary file
faiss_index = faiss.read_index(tmp_file_path)
faiss_index_size = faiss_index.ntotal
print(faiss_index_size)
# Delete the temporary file manually
os.remove(tmp_file_path)
# Create a dataset (if you don't already have one)
dataset = Dataset.from_dict({"embeddings": embeddings_array})
print(len(dataset))
# Add the FAISS index to the dataset
dataset.add_faiss_index("embeddings", custom_index=faiss_index)
print("FAISS index successfully added to the dataset.")
# create_embeddings_io("C://Users//ssoun//AppData//Local//Temp//tmpa27r463c", extractor, model)
