import axios from 'axios';

let image: HTMLImageElement | null = null;
let startX: number = 0;
let startY: number = 0;
let drawing: boolean = false;
let canvas: HTMLCanvasElement | null = null;
let ctx: CanvasRenderingContext2D | null = null;
let rect: { startX: number, startY: number, endX: number, endY: number } = { startX: 0, startY: 0, endX: 0, endY: 0 };

const imageInput: HTMLInputElement = document.getElementById("imageInput") as HTMLInputElement;
const uploadButton: HTMLButtonElement = document.getElementById("uploadButton") as HTMLButtonElement;

imageInput.addEventListener("change", handleImageUpload);
uploadButton.addEventListener("click", (event) => {
  const imageFile = (imageInput.files && imageInput.files[0]) ? imageInput.files[0] : null;
  if (imageFile) {
    uploadImageAndMetadata(imageFile, [rect]);
  } else {
    alert("No image selected.");
  }
});

function handleImageUpload(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      image = new Image();
      image.onload = initializeCanvas;
      image.src = e.target?.result as string;
    };
    reader.readAsDataURL(file);
  }
}

function initializeCanvas() {
  if (!image) return;

  canvas = document.getElementById("canvas") as HTMLCanvasElement;
  ctx = canvas?.getContext("2d")!;
  canvas.width = image.width;
  canvas.height = image.height;

  ctx?.drawImage(image, 0, 0);

  // Enable upload button
  uploadButton.disabled = false;

  // Enable drawing rectangle
  canvas?.addEventListener("mousedown", startDraw);
  canvas?.addEventListener("mousemove", drawRectangle);
  canvas?.addEventListener("mouseup", endDraw);
}

function startDraw(event: MouseEvent) {
  if (canvas && ctx && image) {
    drawing = true;
    startX = event.offsetX;
    startY = event.offsetY;
  }
}

function drawRectangle(event: MouseEvent) {
  if (!drawing || !canvas || !ctx) return;

  const endX = event.offsetX;
  const endY = event.offsetY;
  rect = { startX, startY, endX, endY };

  redrawCanvas();
  ctx.strokeStyle = "red";
  ctx.lineWidth = 2;
  ctx.strokeRect(startX, startY, endX - startX, endY - startY);
}

function endDraw() {
  drawing = false;
}

function redrawCanvas() {
  if (!canvas || !ctx || !image) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0);
}

// Function to get a signed URL from Google Cloud Storage
async function getSignedUrl(filename: string, contentType: string) {
  const response = await axios.post(
    `https://storage.googleapis.com/upload/storage/v1/b/${BUCKET_NAME}/o?uploadType=resumable&name=${filename}`,
    {},
    {
      headers: {
        "Content-Type": contentType,
        Authorization: `Bearer YOUR_ACCESS_TOKEN`, // Replace with actual token
      },
    }
  );
  return response.headers.location;
}

// Function to upload file to GCS
async function uploadToGCS(signedUrl: string, file: Blob) {
  await axios.put(signedUrl, file, {
    headers: { "Content-Type": file.type },
  });
}

// Function to upload the image and rectangle metadata
async function uploadImageAndMetadata(imageFile: File, rectangles: any[]) {
  try {
    // Step 1: Generate unique filenames
    const timestamp = Date.now();
    const imageFilename = `${IMAGE_FOLDER}/${timestamp}.png`;
    const jsonFilename = `${JSON_FOLDER}/${timestamp}.json`;

    // Step 2: Get signed URLs
    const imageSignedUrl = await getSignedUrl(imageFilename, "image/png");
    const jsonSignedUrl = await getSignedUrl(jsonFilename, "application/json");

    // Step 3: Upload Image
    await uploadToGCS(imageSignedUrl, imageFile);

    // Step 4: Upload Rectangle Metadata
    const metadata = JSON.stringify({ imageUrl: `https://storage.googleapis.com/${BUCKET_NAME}/${imageFilename}`, rectangles });
    const metadataBlob = new Blob([metadata], { type: "application/json" });
    await uploadToGCS(jsonSignedUrl, metadataBlob);

    console.log("Upload successful!");
    alert("Image and metadata uploaded successfully!");

  } catch (error) {
    console.error("Upload failed:", error);
    alert("Upload failed.");
  }
}