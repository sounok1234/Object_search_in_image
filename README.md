# Similar_objects_in_image 

POC to find similar objects in an image from an object selection

## Run the app locally: 

uvicorn app:app --reload

## Improvements:

1. Improve scaling for patches for time efficiency 
2. Filter out patches based on color 
3. Improve the UI (add slider for number of patches to show)
4. Free GPU access
5. Single object search across multiple images
6. Support multiple users and super large images 

## Extra libraries

1. python-multipart
2. torch 