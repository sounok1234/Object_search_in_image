import axios from "axios";
import './styles.scss';

const BUCKET_NAME = "faiss-storage-1234";
const IMAGE_FOLDER = "uploads";
const JSON_FOLDER = "metadata";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const upload = document.getElementById("upload") as HTMLInputElement;
const submitBtn = document.getElementById("submit") as HTMLButtonElement;

let img = new Image();
let drawing = false;
let startX = 0, startY = 0, endX = 0, endY = 0;
let rectangles: { x: number; y: number; width: number; height: number }[] = [];
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

// Draw image onto canvas after loading
img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
};

// Start drawing rectangle
canvas.addEventListener("mousedown", (event) => {
    drawing = true;
    startX = event.offsetX;
    startY = event.offsetY;
});

// Draw rectangle dynamically
canvas.addEventListener("mousemove", (event) => {
    if (!drawing) return;
    
    endX = event.offsetX;
    endY = event.offsetY;

    // Redraw everything
    redrawCanvas();
    
    // Draw new rectangle
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, endX - startX, endY - startY);
});

// Save rectangle when mouse released
canvas.addEventListener("mouseup", () => {
    if (drawing) {
        rectangles.push({
            x: startX,
            y: startY,
            width: endX - startX,
            height: endY - startY
        });
    }
    drawing = false;
});

// Redraw canvas (image + stored rectangles)
function redrawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    rectangles.forEach(rect => {
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
    });
}

// Upload image + metadata to GCS
submitBtn.addEventListener("click", async () => {
    if (!imageFile || rectangles.length === 0) {
        alert("Upload an image and draw at least one rectangle.");
        return;
    }

    try {
        // Step 1: Generate unique filenames
        const uuid = crypto.randomUUID(); // Generates a unique ID

        // Step 2: Prepare FormData
        const formData = new FormData();
        formData.append("file", imageFile); // Append the image
        formData.append("uuid", uuid); // Append the unique identifier
        formData.append("rectangles", JSON.stringify(rectangles)); // Append rectangle data

        // Step 3: Send to FastAPI backend
        const response = await fetch("https://object-search-961989467540.us-central1.run.app/upload/", {
            method: "POST",
            body: formData
        });

        // Step 4: Handle the response
        const result = await response.json();
        if (response.ok) {
            alert("Image uploaded successfully! UUID: " + uuid);
            rectangles = []; // Clear rectangles after upload
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