import os
from datasets import load_dataset
from google.cloud import storage

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

def upload_to_gcs(source_file_dir, destination_blob_name):
    """Uploads a file to Google Cloud Storage."""
    blob = bucket.blob(destination_blob_name)
    for root, _, files in os.walk(source_file_dir):
        for file_name in files:
            blob.upload_from_filename(os.path.join(root, file_name))
    print(f"Uploaded {source_file_dir} to {destination_blob_name}")

def download_from_gcs(blob_name, destination_file_name):
    """Downloads a file from Google Cloud Storage."""
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {blob_name} to {destination_file_name}")

def extract_embeddings(image, extractor, model):
    image_pp = extractor(image, return_tensors="pt")
    features = (model(**image_pp).last_hidden_state[:, 0].detach().numpy()).squeeze()
    return features.squeeze()

def create_embeddings(output_dir, extractor, model):    
    dataset = load_dataset("imagefolder", data_dir=output_dir)
    train_dataset = dataset['train']    
    dataset_with_embeddings = train_dataset.map(lambda example: 
                            {'embeddings': extract_embeddings(example["image"], extractor, model)})
    # Save dataset (temporarily) and upload to GCS
    dataset_with_embeddings.save_to_disk("/tmp/old_embeddings")
    upload_to_gcs("/tmp/old_embeddings", GCS_EMBEDDINGS_FILE)
    # Create FAISS index
    dataset_with_embeddings.add_faiss_index(column="embeddings")
    # Save FAISS index to GCS
    dataset_with_embeddings.save_faiss_index("embeddings", "/tmp/old_index.faiss")
    upload_to_gcs("/tmp/old_index.faiss", GCS_FAISS_INDEX_FILE)
    # return dataset_with_embeddings
    return {"message": "Embeddings saved & uploaded to GCS!"}

def get_neighbors(extractor, model, dataset_with_embeddings, query_image, top_k):
    qi_embedding = model(**extractor(query_image, return_tensors="pt"))
    qi_embedding = qi_embedding.last_hidden_state[:, 0].detach().numpy().squeeze()
    scores, retrieved_examples = dataset_with_embeddings.get_nearest_examples('embeddings', qi_embedding, k=top_k)
    return scores, retrieved_examples

