import cv2
from shapely.geometry import box as shapely_box
from rtree import index
import numpy as np


def remove_overlaps_spatial(boxes):
    idx = index.Index()
    selected = []
    
    for i, b in enumerate(boxes):
        b_geom = shapely_box(*b)
        hits = list(idx.intersection(b, objects=True))
        
        if not any(shapely_box(*boxes[hit.id]).intersects(b_geom) for hit in hits):
            idx.insert(i, b)
            selected.append(b)
    
    return selected

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
        boxes_cv2 = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in boxes]
        indices = cv2.dnn.NMSBoxes(boxes_cv2, normalized_scores, score_threshold=0.0, nms_threshold=0.4)
        filtered_boxes = [boxes[i] for i in indices]
        final_filter = remove_overlaps_spatial(filtered_boxes)
        return final_filter
