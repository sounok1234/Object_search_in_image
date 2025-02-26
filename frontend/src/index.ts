import './styles.scss';
import { v4 as uuidv4 } from 'uuid';

// DOM Elements
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const upload = document.getElementById("upload") as HTMLInputElement;
const submitBtn = document.getElementById("submit") as HTMLButtonElement;
const sidebar = document.getElementById("sidebar") as HTMLElement;
const numInput = document.getElementById("numInput") as HTMLInputElement | null;

// Backend URL and user requests map
const url = process.env.BACKEND_URL || "";
const userRequests = new Map<string, boolean>();

// Image and Canvas Properties
let img = new Image();
let imageFile: File | null = null;
let imageWidth = 0, imageHeight = 0;
let canvasWidth = 0, canvasHeight = 0;

// Drawing state
let drawing = false;
let startX = 0, startY = 0, endX = 0, endY = 0;
let rectangle: { x: number; y: number; width: number; height: number } | null = null;

/** 
 * Adjusts canvas size dynamically 
 */
function resizeCanvas(): void {
    const sidebarWidth = sidebar.offsetWidth;
    canvas.width = window.innerWidth - sidebarWidth;
    canvas.height = window.innerHeight;
    canvasWidth = canvas.width;
    canvasHeight = canvas.height;
    resizeImageToFitCanvas();
}

/** 
 * Resizes and centers the image on the canvas while maintaining aspect ratio
 */
function resizeImageToFitCanvas(): void {
    const scale = Math.min(canvasWidth / imageWidth, canvasHeight / imageHeight);
    const scaledWidth = imageWidth * scale;
    const scaledHeight = imageHeight * scale;
    const offsetX = (canvasWidth - scaledWidth) / 2;
    const offsetY = (canvasHeight - scaledHeight) / 2;
    redrawCanvas(offsetX, offsetY, scaledWidth, scaledHeight);
}

/**
 * Redraws the image and any existing rectangles
 */
function redrawCanvas(offsetX: number, offsetY: number, scaledWidth: number, scaledHeight: number): void {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);
    if (rectangle) {
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(rectangle.x, rectangle.y, rectangle.width, rectangle.height);
    }
}

/**
 * Draws the detected object rectangles from backend response
 */
function drawDetectedRectangles(boxes: number[][], indices: number[], numOfObj: number): void {
    const scaleX = canvasWidth / imageWidth;
    const scaleY = canvasHeight / imageHeight;
    const numOfObjects = indices.slice(0, numOfObj);

    numOfObjects.forEach((index) => {
        if (index >= 0 && index < boxes.length) {
            const [x1, y1, x2, y2] = boxes[index];

            const x = x1 * scaleX;
            const y = y1 * scaleY;
            const width = (x2 - x1) * scaleX;
            const height = (y2 - y1) * scaleY;

            ctx.strokeStyle = "green";  // Set stroke color to green
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, width, height);
        }
    });
}

/**
 * Gets the mouse position adjusted for canvas scaling
 */
function getMousePos(event: MouseEvent): { x: number; y: number } {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (event.clientX - rect.left) * scaleX,
        y: (event.clientY - rect.top) * scaleY
    };
}

/**
 * Handles image upload and loads it into the canvas
 */
function handleImageUpload(event: Event): void {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) {
        imageFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            img.src = e.target?.result as string;
        };
        reader.readAsDataURL(file);
    }
}

/**
 * Uploads the image and rectangle metadata to the backend
 */
async function handleSubmit(): Promise<void> {
    if (!imageFile || !rectangle) {
        alert("Upload an image and draw a rectangle.");
        return;
    }

    try {
        const uuid = uuidv4();
        userRequests.set(uuid, true);

        const x = Math.round((rectangle.x / canvasWidth) * imageWidth);
        const y = Math.round((rectangle.y / canvasHeight) * imageHeight);
        const adjustedRectangle = {
            x1: x,
            y1: y,
            x2: Math.round((rectangle.width / canvasWidth) * imageWidth + x),
            y2: Math.round((rectangle.height / canvasHeight) * imageHeight + y)
        };

        const formData = new FormData();
        formData.append("file", imageFile);
        formData.append("uuid", uuid);
        formData.append("rectangle", JSON.stringify(adjustedRectangle));

        const response = await fetch(url, { method: "POST", body: formData });
        const result = await response.json();

        if (userRequests.has(result.uuid)) {
            console.log("Response for UUID:", result.uuid, "->", result);
            drawDetectedRectangles(result['boxes'], result['indices'], Number(numInput?.value));
            userRequests.delete(result.uuid);
        }
    } catch (error) {
        console.error("Upload failed:", error);
        alert("Upload failed.");
    }
}

// Event Listeners

// Image load event to adjust canvas size
img.onload = () => {
    imageWidth = img.naturalWidth;
    imageHeight = img.naturalHeight;
    resizeCanvas();
};

// Mouse events for drawing a rectangle
canvas.addEventListener("mousedown", (event: MouseEvent) => {
    drawing = true;
    const pos = getMousePos(event);
    startX = pos.x;
    startY = pos.y;
});

canvas.addEventListener("mousemove", (event: MouseEvent) => {
    if (!drawing) return;
    const pos = getMousePos(event);
    endX = pos.x;
    endY = pos.y;
    
    rectangle = {
        x: Math.min(startX, endX),
        y: Math.min(startY, endY),
        width: Math.abs(endX - startX),
        height: Math.abs(endY - startY)
    };
    resizeImageToFitCanvas();
});

canvas.addEventListener("mouseup", () => drawing = false);

// Attach event listeners
upload.addEventListener("change", handleImageUpload);
submitBtn.addEventListener("click", handleSubmit);
window.addEventListener("resize", resizeCanvas);
