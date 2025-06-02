import numpy as np
from torchvision.ops import nms
import torch


class DatasetWithEmbeddings:
    def __init__(self, extractor, model):
        self.extractor = extractor
        self.model = model

    def extract_embeddings(self, image):
        image_pp = self.extractor(image, return_tensors="pt")
        features = (self.model(**image_pp).last_hidden_state[:, 0].detach().numpy()).squeeze()
        return features.squeeze()

    def get_neighbors(self, dataset_with_embeddings, query_image, top_k):
        query_embedding = self.model(**self.extractor(query_image, return_tensors="pt"))
        query_embedding = query_embedding.last_hidden_state[:, 0].detach().numpy().squeeze()
        scores, retrieved_examples = dataset_with_embeddings.get_nearest_examples('embeddings', query_embedding, k=top_k)
        # return scores, retrieved_examples  
        coords = np.array(retrieved_examples['coords'])
        min_val, max_val = np.min(scores), np.max(scores)
        normalized_scores = [(1 - (s - min_val) / (max_val - min_val)) for s in scores]
        # Get boxes from coords
        boxes = [[c[0][0], c[0][1], c[1][0], c[1][1]] for c in coords.tolist()]
        # Apply Non-Maximum Suppression (NMS)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(normalized_scores, dtype=torch.float32)
        filtered_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.3).tolist()
        filtered_boxes = [boxes[i] for i in filtered_indices]
        return filtered_boxes
