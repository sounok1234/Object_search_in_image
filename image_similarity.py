from transformers import AutoFeatureExtractor, AutoModel
from datasets import load_dataset

image_path = "Plan2.png"
output_dir = "patches_train"
model_ckpt = "google/vit-base-patch16-224"

extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
hidden_dim = model.config.hidden_size

dataset = load_dataset("imagefolder", data_dir=output_dir)
train_dataset = dataset['train']

def extract_embeddings(image):
    image_pp = extractor(image, return_tensors="pt")
    features = (model(**image_pp).last_hidden_state[:, 0].detach().numpy()).squeeze()
    return features.squeeze()

def create_embeddings():
    dataset_with_embeddings = train_dataset.map(lambda example: {'embeddings': extract_embeddings(example["image"])})
    # save dataset with embeddings variable to disk
    dataset_with_embeddings.save_to_disk('old_embeddings')
    # load dataset with embeddings from disk
    dataset_with_embeddings.add_faiss_index(column='embeddings')
    # save the faiss index to disk
    dataset_with_embeddings.save_faiss_index('embeddings', 'old_index.faiss')
    return dataset_with_embeddings

def get_neighbors(dataset_with_embeddings, query_image, top_k):
    qi_embedding = model(**extractor(query_image, return_tensors="pt"))
    qi_embedding = qi_embedding.last_hidden_state[:, 0].detach().numpy().squeeze()
    scores, retrieved_examples = dataset_with_embeddings.get_nearest_examples('embeddings', qi_embedding, k=top_k)
    return scores, retrieved_examples

