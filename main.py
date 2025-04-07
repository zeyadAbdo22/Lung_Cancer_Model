from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import requests

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

# Download the model if it doesn't exist
MODEL_PATH = "lung_cancer_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1l4WjZAdPL-6XqWwwSXXv8lvss5YigxaE"

if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model from Google Drive...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded successfully.")

# Load model
model = load_model(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "✅ Lung Cancer Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

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
