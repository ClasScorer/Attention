import os
import cv2
import torch
import numpy as np
from PIL import Image
import io
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from enum import Enum
import logging
from typing import List, Optional, Dict, Any, Set
import sys
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
from datetime import datetime
from prisma import Prisma
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration management
class Settings(BaseSettings):
    app_name: str = "Attention Detection API"
    version: str = "1.0.0"
    debug: bool = bool(os.getenv("APP_DEBUG", "False") == "True")
    allowed_origins: Set[str] = {"*"}
    max_image_size: int = int(os.getenv("APP_MAX_IMAGE_SIZE", str(10 * 1024 * 1024)))  # Default 10MB
    inout_threshold: float = float(os.getenv("APP_INOUT_THRESHOLD", "0.5"))
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/db")
    
    class Config:
        env_prefix = "APP_"
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
limiter = Limiter(key_func=get_remote_address)
prisma = Prisma()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize the model globally to avoid reloading
model = None
transform = None
device = None

def initialize_model():
    """Load and prepare the Gazelle model"""
    global model, transform, device
    if model is None:
        logger.info("Initializing Gazelle model...")
        model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        model.to(device)
        logger.info(f"Model initialized on {device}")
    return model, transform, device

# Enhanced response models
class AttentionStatus(str, Enum):
    """
    Possible results from attention detection analysis.
    
    Values:
        FOCUSED: Person is looking at the camera or nearby
        UNFOCUSED: Person is looking away from the camera
        NO_FACE_DETECTED: No faces were detected in the image
    """
    FOCUSED = "FOCUSED"
    UNFOCUSED = "UNFOCUSED"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"

# Simplified input and output models for cropped face images
class FaceAttentionRequest(BaseModel):
    """Input model for attention detection with face ID"""
    face_id: str = Field(
        description="Unique identifier for the face", 
        example="user123"
    )
    lecture_id: str = Field(
        description="Unique identifier for the lecture",
        example="lecture123"
    )
    timestamp: datetime = Field(
        description="Timestamp of the attention check",
        example="2025-03-06T23:17:05.664Z"
    )

class FaceAttentionResponse(BaseModel):
    """
    Response model for face attention detection results
    """
    status: str = Field(
        description="Status of the request processing",
        example="success"
    )
    face_id: str = Field(
        description="The face ID provided in the request",
        example="user123"
    )
    attention_status: AttentionStatus = Field(
        description="Whether the person is focused on the camera",
        example="focused"
    )
    confidence: float = Field(
        description="Confidence score for the detection",
        example=0.95
    )

class ErrorResponse(BaseModel):
    """
    Response model for error cases.
    """
    status: str = Field(
        description="Error status",
        example="error"
    )
    message: str = Field(
        description="Detailed error message",
        example="File size exceeds maximum allowed size of 10MB"
    )

# API Documentation
API_DESCRIPTION = """
# Attention Detection API

This API analyzes cropped face images to determine if a person is paying attention.

## Features

- Simple focus/unfocus detection
- Works with pre-cropped face images
- Returns attention status with confidence score

## Usage Limits

- Maximum file size: 10MB
- Rate limit: 5 requests per minute per IP
- Supported image formats: JPEG, PNG

## Input Requirements

- Images must be pre-cropped to the face region
- Each request should include a face ID

## Output

- Attention status (focused/unfocused)
- Confidence score

## Error Codes

- 400: Bad Request (Invalid input)
- 413: Payload Too Large
- 429: Too Many Requests
- 500: Internal Server Error
"""

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description=API_DESCRIPTION,
    version=settings.version,
    debug=settings.debug,
    openapi_tags=[
        {
            "name": "Detection",
            "description": "Endpoints for attention detection"
        },
        {
            "name": "Health",
            "description": "Endpoints for API health monitoring"
        }
    ],
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestValidationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Add request ID and timestamp
        request.state.request_id = f"{int(time.time())}_{id(request)}"
        request.state.start_time = datetime.utcnow()
        
        # Log incoming request
        logger.info(
            f"Request started - ID: {request.state.request_id} "
            f"Method: {request.method} Path: {request.url.path}"
        )
        
        response = await call_next(request)
        
        # Calculate and add processing time header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request.state.request_id
        
        # Log response
        logger.info(
            f"Request completed - ID: {request.state.request_id} "
            f"Status: {response.status_code} "
            f"Processing Time: {process_time:.3f}s"
        )
        
        return response

# Add middleware to app
app.add_middleware(RequestValidationMiddleware)

def determine_focus_status(inout_score: float, is_centered: bool) -> AttentionStatus:
    """Determine if the person is focused based on gaze vector length"""
    # For a cropped face image, we use a more constrained threshold since the face fills more of the image
    return AttentionStatus.FOCUSED if inout_score > 0.03 or is_centered else AttentionStatus.UNFOCUSED

async def process_face_image(image):
    """
    Process a cropped face image with the Gazelle model and analyze attention
    
    Args:
        image: PIL Image object containing a cropped face
        
    Returns:
        Dictionary with attention analysis results
    """
    try:
        # Ensure model is initialized
        model, transform, device = initialize_model()
        
        # For a cropped face image, we use the entire image as the bounding box [0,0,1,1]
        norm_bbox = [[0.0, 0.0, 1.0, 1.0]]

        # Prepare input
        img_tensor = transform(image).unsqueeze(0).to(device)
        model_input = {
            "images": img_tensor,
            "bboxes": [norm_bbox]
        }

        # Run inference
        with torch.no_grad():
            output = model(model_input)

        heatmap = output['heatmap'][0][0]
        inout_score = output['inout'][0][0].item() if output['inout'] is not None else None

        # Get heatmap center
        heatmap_np = heatmap.detach().cpu().numpy()
        heatmap_y, heatmap_x = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
        heatmap_center_x = heatmap_x / heatmap_np.shape[1]
        heatmap_center_y = heatmap_y / heatmap_np.shape[0]

        # Define threshold for "centered" gaze
        threshold = 0.4
        is_centered = (
            abs(heatmap_center_x - 0.5) < threshold and
            abs(heatmap_center_y - 0.5) < threshold
        )

        
        # Determine focus status
        focus_status = determine_focus_status(inout_score, is_centered)
        
        # Create response
        response = {
            "status": "success",
            "attention_status": focus_status.value,
            "confidence": float(inout_score)
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing face image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup():
    """Initialize the model and database connection on startup"""
    initialize_model()
    await prisma.connect()

@app.on_event("shutdown")
async def shutdown():
    """Close database connection on shutdown"""
    await prisma.disconnect()

@app.post("/detect-face-attention", 
         response_model=FaceAttentionResponse,
         responses={
             400: {"model": ErrorResponse, "description": "Invalid input"},
             413: {"model": ErrorResponse, "description": "File too large"},
             429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
             500: {"model": ErrorResponse, "description": "Internal server error"},
         },
         tags=["Detection"],
         summary="Detect if a person is paying attention from a face image",
         description="""
Upload a cropped face image to determine if the person is focused or unfocused.

This endpoint analyzes a pre-cropped face image and returns whether the person is paying 
attention, with a binary focused/unfocused result.

**Input Requirements:**
- Image must be a pre-cropped face image in JPEG or PNG format
- Face ID must be provided to identify the person
- Maximum image size: 10MB

**Response includes:**
- The provided face ID
- Attention status (focused/unfocused)
- Confidence score for the detection

**Example Usage:**
```python
import requests

# Prepare image
with open("face.jpg", "rb") as image_file:
    image_data = image_file.read()

# Make request
url = "http://localhost:8000/detect-face-attention"
files = {"file": ("face.jpg", image_data, "image/jpeg")}
data = {"face_id": "user123"}
response = requests.post(url, files=files, data=data)

# Process response
result = response.json()
print(f"Face ID: {result['face_id']}")
print(f"Attention: {result['attention_status']}")
print(f"Confidence: {result['confidence']:.2f}")
```
         """)
@limiter.limit(os.getenv("APP_RATE_LIMIT", "5/minute"))
async def detect_face_attention(
    request: Request, 
    file: UploadFile = File(...),
    face_id: str = Form(...),
    lecture_id: str = Form(...),
    timestamp: str = Form(...)
) -> FaceAttentionResponse:
    """
    Detect attention from a cropped face image.
    
    Args:
        file: Uploaded face image file
        face_id: Unique identifier for the face
        
    Returns:
        FaceAttentionResponse: Detection results
        
    Raises:
        HTTPException: For invalid input or processing errors
    """
    try:
        logger.info(f"Processing face attention detection - Request ID: {request.state.request_id}")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            logger.warning(f"Invalid file type: {file.content_type} - Request ID: {request.state.request_id}")
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Validate file size
        contents = await file.read()
        file_size = len(contents)
        logger.info(f"File size: {file_size/1024:.2f}KB - Request ID: {request.state.request_id}")
        
        if file_size > settings.max_image_size:
            logger.warning(f"File too large: {file_size/1024/1024:.2f}MB - Request ID: {request.state.request_id}")
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed size of {settings.max_image_size // (1024 * 1024)}MB"
            )
        
        # Process image
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to decode image: {str(e)} - Request ID: {request.state.request_id}")
            raise HTTPException(
                status_code=400,
                detail="Invalid image file or format"
            )
        
        # Check image dimensions
        width, height = image.size
        if width < 64 or height < 64:
            logger.warning(f"Image too small: {width}x{height} - Request ID: {request.state.request_id}")
            raise HTTPException(
                status_code=400,
                detail="Image dimensions too small. Minimum size is 64x64 pixels."
            )
        
        # Process image with model
        result = await process_face_image(image)
        logger.info(f"Processed face image - Face ID: {face_id} - Request ID: {request.state.request_id}")
        
        # Parse timestamp
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid timestamp format. Use ISO format (e.g., 2025-03-06T23:17:05.664Z)"
            )

        # Store result in database using the Prisma enum
        # try:
        #     await prisma.attentionschema.create({
        #         'studentId': face_id,
        #         'lectureId': lecture_id,
        #         'timestamp': parsed_timestamp,
        #         'attentionStatus': result['attention_status'],
        #         'confidence': result['confidence']
        #     })
        # except Exception as e:
        #     logger.error(f"Database error: {str(e)} - Request ID: {request.state.request_id}")
        #     raise HTTPException(
        #         status_code=500,
        #         detail="Failed to store attention data in database"
        #     )

        # Add face_id to result
        result["face_id"] = face_id
        return FaceAttentionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)} - Request ID: {request.state.request_id}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image processing"
        )

@app.get("/",
         response_model=dict,
         tags=["Health"],
         summary="Health check endpoint")
async def root() -> dict:
    """Root endpoint to verify the API is running."""
    return {
        "status": "ok",
        "message": "Attention detection API is running",
        "version": settings.version
    }

if __name__ == "__main__":
    import uvicorn
    # Initialize model at startup
    initialize_model()
    uvicorn.run(app, host="0.0.0.0", port=23123) 