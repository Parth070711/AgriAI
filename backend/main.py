import os
import shutil
from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from models.plant_disease.disease_detector import predict_disease

# Create uploads folder automatically
os.makedirs("uploads", exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://agri-ai-lime-seven.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "message": "AgriAI Backend Running"
    }

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:

        file_path = f"uploads/{file.filename}"

        # Save uploaded image
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Compress and resize image
        image = Image.open(file_path)

        image = image.convert("RGB")

        image.thumbnail((224, 224))

        image.save(
            file_path,
            format="JPEG",
            optimize=True,
            quality=75
        )

        # Predict disease
        result = predict_disease(file_path)

        # Delete image after prediction
        if os.path.exists(file_path):
            os.remove(file_path)

        return result

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }