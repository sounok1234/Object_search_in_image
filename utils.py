def extract_embeddings(image, extractor, model):
    image_pp = extractor(image, return_tensors="pt")
    features = (model(**image_pp).last_hidden_state[:, 0].detach().numpy()).squeeze()
    return features.squeeze()

def get_neighbors(extractor, model, dataset_with_embeddings, query_image, top_k):
    query_embedding = model(**extractor(query_image, return_tensors="pt"))
    query_embedding = query_embedding.last_hidden_state[:, 0].detach().numpy().squeeze()
    scores, retrieved_examples = dataset_with_embeddings.get_nearest_examples('embeddings', query_embedding, k=top_k)
    return scores, retrieved_examples

