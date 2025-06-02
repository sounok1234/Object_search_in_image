# Similar_objects_in_image 

POC to find similar objects in an image from an object selection

## Run the backend app locally: 

docker-compose up --build

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

# Backend
1. Improve scaling for patches for time efficiency (and super large images)
2. Filter out patches based on color 
3. Add GPU access
4. Single object search across multiple images
5. Signed url for images
6. Add task queue to accomodate concurrency
7. Give option for different models

# Frontend
1. Improve the UI (add slider for number of objects to show)
2. Add save option for detected patches and image
3. Add user login and authentication
4. Add undo/redo for drawing patch
5. Add user guides
6. Use WebSockets for Real-Time Responses
7. Same as backend no 7
