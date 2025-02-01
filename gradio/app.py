import cv2
import gradio as gr
import numpy as np
from image_similarity import get_neighbors, create_embeddings
from patches import split_image_into_patches
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def select_patch(original_img: np.ndarray,
                 sel_pix: list,
                 evt: gr.SelectData):
  """When the user clicks on the image, show points and update the mask.

  Args:
      original_img (np.ndarray): Input image.
      sel_pix (list): List of selected points.
      evt (gr.SelectData): Event data.

  """
  img = original_img.copy()
  sel_pix.append(evt.index)

  for point in sel_pix:
    cv2.circle(img, (point[0], point[1]), 10, (0, 0, 255), -1)

  return img, sel_pix

def reset_image(original_image):
  output_dir = "patches_train"
  split_image_into_patches(original_image, output_dir)
  dataset_with_embeddings = create_embeddings()

  print('Image ready!')
  return original_image, dataset_with_embeddings

def normalize_scores(scores):
  min_val = np.min(scores)
  max_val = np.max(scores)
  normalized_min_max = (scores - min_val) / (max_val - min_val)
  subtracted_list = [1- item for item in normalized_min_max]
  return subtracted_list

def get_nms_boxes(coords):
  boxes = []
  for coord in coords:
    corners = [coord[0][0], coord[0][1], coord[1][0], coord[1][1]]
    boxes.append(corners)

  return boxes

def find_similar_objects(dataset_with_embeddings, image, selected_points):
  output_image = image.copy()
  if len(selected_points) == 2:
    point1, point2 = selected_points  
    selected_patch = output_image[point1[1]:point2[1], point1[0]:point2[0]]
    query_patch = cv2.resize(selected_patch , (224, 224))
    scores, retrieved_examples = get_neighbors(dataset_with_embeddings, query_patch, 50)
    coords = np.array(retrieved_examples['coords'])

    normalized_scores = normalize_scores(scores)
    boxes = get_nms_boxes(coords.tolist())
    indices = cv2.dnn.NMSBoxes(boxes, normalized_scores, score_threshold=0.000001, nms_threshold=0.49)

    for i in indices:
      x, y, x2, y2 = boxes[i]
      cv2.rectangle(output_image, (x, y), (x2, y2), (0,0,255), 2)

  cv2.imwrite('output.jpg', output_image)
  return output_image

with gr.Blocks() as demo:
  # store original image without points, default None
  selected_points = gr.State(value=[])
  original_image = gr.State(value=None)
  query_image = gr.State(value=None)
  dataset_with_embeddings = gr.State(value=None)  
  # title
  with gr.Row():
    gr.Markdown("# Similar objects \n"
                "Find objects similar to the one you have boxed out")

  # Segment image
  with gr.Row():
    with gr.Column():
      # input image
      input_image = gr.Image(type="numpy", label='Input image')
      # Set height of widget to 500 pixels
      input_image.style(height=600)
      similar_button = gr.Button('Find similar objects')

    with gr.Column():
      output_image = gr.Image(type="numpy", label='Output image')
      output_image.style(height=800, width=800)

  input_image.upload(
    reset_image,
    [input_image],
    [original_image, dataset_with_embeddings]      
  )

  input_image.select(
    select_patch,
    [original_image, selected_points],
    [input_image, selected_points])

  similar_button.click(
    find_similar_objects,
    [dataset_with_embeddings, input_image, selected_points],
    [output_image]
  )

demo.queue().launch(debug=True, enable_queue=True)
