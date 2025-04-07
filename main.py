from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware
import os
import requests

# Define path to model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "lung_cancer_model.h5")

# Download the model from Google Drive if it doesn't exist
if not os.path.exists(model_path):
    print("📥 Downloading model from Google Drive...")
    url = "https://drive.google.com/uc?export=download&id=1l4WjZAdPL-6XqWwwSXXv8lvss5YigxaE"  # Direct download link
    r = requests.get(url)
    if r.status_code == 200:
        with open(model_path, "wb") as f:
            f.write(r.content)
        print("✅ Model downloaded successfully.")
    else:
        raise Exception(f"❌ Failed to download model. Status code: {r.status_code}")

# Load the model
model = load_model(model_path)

# Class labels the model can predict
CLASS_NAMES = ["Adenocarcinoma", "Benign", "Squamous Cell Carcinoma"]

# Initialize FastAPI app
app = FastAPI()

# Enable CORS to allow frontend apps (like React/Streamlit) to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Image size expected by the model
IMG_SIZE = 224

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "✅ Lung Cancer Detection API is running!"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the uploaded image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Run prediction
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        predicted_label = CLASS_NAMES[predicted_class]
        confidence = float(np.max(prediction))

        # Return prediction result
        return JSONResponse(content={
            "prediction_raw": prediction.tolist(),
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        # Handle errors gracefully
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the app (Railway-friendly)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
