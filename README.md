# Similar_objects_in_image 

POC to find similar objects in an image from an object selection

## Run the backend app locally: 

docker-compose up --build

## Deploy backend: 

/backend gcloud builds submit --tag gcr.io/<project-id>/backend
gcloud run deploy backend --image gcr.io/<project-id>/backend --platform managed --region us-central1 --allow-unauthenticated

## Run the frontend app locally:

1. Install the required node modules:

```
npm install
```

2. Run the Typescript server to preview

```
npm run serve
```

3. And finally build into `public`

```
npm run build
```

## Improvements:

1. Improve scaling for patches for time efficiency 
2. Filter out patches based on color 
3. Improve the UI (add slider for number of patches to show)
   Add save option for image and detected patches
4. Free GPU access
5. Single object search across multiple images
6. Support multiple users and super large images 
