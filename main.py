from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import uvicorn
import os
import io
import gdown

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "lung_cancer_model.h5")

# Download the model from Google Drive if it's not already present
if not os.path.exists(model_path):
    print("📥 Downloading model from Google Drive...")
    url = "https://drive.google.com/uc?id=1l4WjZAdPL-6XqWwwSXXv8lvss5YigxaE"  # Direct download link
    gdown.download(url, model_path, quiet=False)
    print("✅ Model downloaded successfully.")

# Load the model
model = load_model(model_path)

# Class names the model predicts
CLASS_NAMES = ["Adenocarcinoma", "Benign", "Squamous Cell Carcinoma"]

# Initialize the FastAPI app
app = FastAPI()

# Allow connections from any frontend (e.g. Streamlit, React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model expects image size 224x224
IMG_SIZE = 224

# Health check route
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
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))
        predicted_label = CLASS_NAMES[predicted_class]
        confidence = float(np.max(prediction))

        # Return prediction
        return JSONResponse(content={
            "prediction_raw": prediction.tolist(),
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the server (used when running locally, or by Railway)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
