from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import os
import requests
import kagglehub


app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ["Adenocarcinoma", "Benign", "Squamous Cell Carcinoma"]
IMG_SIZE = 224


model = None  # Global model variable

def load_model_from_kaggle():
    """Load model from Kaggle Hub"""
    global model
    path = kagglehub.model_download("zeyadabdo/lung-cancer-resnet/keras/v1")
    model_path = os.path.join(path, "lung-cancer-resnet-model.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found.")

    model = load_model(model_path, compile=False)
    print("✅ Model loaded from Kaggle.")
    return model

@app.on_event("startup")
async def startup_event():
    """Load model at startup"""
    try:
        load_model_from_kaggle()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

@app.get("/")
def root():
    return {"message": "✅ Lung Cancer Detection API is running!"}

@app.post("/lung-cancer")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # Convert the image to array and preprocess for ResNet
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # <-- Important for ResNet

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        predicted_label = CLASS_NAMES[predicted_class]
        confidence = float(np.max(prediction))

        return JSONResponse(content={
            "prediction_raw": prediction.tolist(),
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

