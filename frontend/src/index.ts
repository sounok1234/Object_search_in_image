import axios from "axios";
import './styles.scss';
import { v4 as uuidv4 } from 'uuid';

const BUCKET_NAME = "faiss-storage-1234";
const IMAGE_FOLDER = "uploads";
const JSON_FOLDER = "metadata";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")! as CanvasRenderingContext2D;
const upload = document.getElementById("upload") as HTMLInputElement;
const submitBtn = document.getElementById("submit") as HTMLButtonElement;

let img = new Image();
let drawing = false;
let startX = 0, startY = 0, endX = 0, endY = 0;
let rectangle: { x: number; y: number; width: number; height: number } | null = null;
let imageFile: File | null = null;

// Handle image upload
upload.addEventListener("change", (event) => {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) {
        imageFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            img.src = e.target?.result as string;
        };
        reader.readAsDataURL(file);
    }
});

const numInput = document.getElementById("numInput") as HTMLInputElement | null;
if (numInput) {
    // Ensure only integers are allowed
    numInput.addEventListener("input", () => {
        numInput.value = Math.round(Number(numInput.value)).toString();
    });
}

// Ensure canvas matches image size
img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;

    // Apply the same size to prevent CSS scaling issues
    canvas.style.width = `${img.width}px`;
    canvas.style.height = `${img.height}px`;

    ctx.drawImage(img, 0, 0);
};

// Function to get correct mouse position, adjusting for canvas scaling
function getMousePos(event: MouseEvent): { x: number; y: number } {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY
    };
}

// Start drawing rectangle
canvas.addEventListener("mousedown", (event: MouseEvent) => {
    drawing = true;
    const pos = getMousePos(event);
    startX = pos.x;
    startY = pos.y;
});

// Draw rectangle dynamically
canvas.addEventListener("mousemove", (event: MouseEvent) => {
    if (!drawing) return;

    const pos = getMousePos(event);
    endX = pos.x;
    endY = pos.y;

    // Calculate rectangle properties
    const x = Math.min(startX, endX);
    const y = Math.min(startY, endY);
    const width = Math.abs(endX - startX);
    const height = Math.abs(endY - startY);

    // Redraw everything
    redrawCanvas();

    // Draw new rectangle
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
});

// Save rectangle when mouse released
canvas.addEventListener("mouseup", () => {
    if (drawing) {
        const x = Math.min(startX, endX);
        const y = Math.min(startY, endY);
        const width = Math.abs(endX - startX);
        const height = Math.abs(endY - startY);

        rectangle = { x, y, width, height };
    }
    drawing = false;
});

// Function to redraw the canvas (image + stored rectangles)
function redrawCanvas(): void {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    if (rectangle !== null) {
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
    }
}

// Upload image + metadata to GCS
submitBtn.addEventListener("click", async () => {
    if (!imageFile || rectangle === null) {
        alert("Upload an image and d|raw at least one rectangle.");
        return;
    }

    try {
        // Step 1: Generate unique filenames
        const uuid = uuidv4(); // Generates a unique ID

        // Step 2: Prepare FormData
        const formData = new FormData();
        formData.append("file", imageFile); // Append the image
        formData.append("uuid", uuid); // Append the unique identifier
        formData.append("rectangles", JSON.stringify(rectangle)); // Append rectangle data

        // Step 3: Send to FastAPI backend
        const response = await fetch("https://object-search-961989467540.us-central1.run.app/upload/", {
            method: "POST",
            body: formData
        });

        // Step 4: Handle the response
        const result = await response.json();
        if (response.ok) {
            alert("Image uploaded successfully! UUID: " + uuid);
            rectangle = null; // Clear rectangles after upload
            redrawCanvas(); // Reset canvas
        } else {
            throw new Error(result.error || "Upload failed.");
        }
    } catch (error) {
        console.error("Upload failed:", error);
        alert("Upload failed.");
    }
});

// Get a signed URL from GCS
// async function getSignedUrl(filename: string, contentType: string) {
//     const response = await axios.post(
//         `https://storage.googleapis.com/upload/storage/v1/b/${BUCKET_NAME}/o?uploadType=resumable&name=${filename}`,
//         {},
//         {
//             headers: {
//                 "Content-Type": contentType,
//                 Authorization: `Bearer YOUR_ACCESS_TOKEN`, // Replace with actual token
//             },
//         }
//     );
//     return response.headers.location;
// }

// // Upload file to GCS
// async function uploadToGCS(signedUrl: string, file: Blob) {
//     await axios.put(signedUrl, file, {
//         headers: { "Content-Type": file.type },
//     });
// }