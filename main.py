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
import kagglehub
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
IMG_SIZE = 224
LUNG_CLASSES = ["Adenocarcinoma", "Benign", "Squamous Cell Carcinoma"]
BRAIN_CLASSES = {0: "Normal", 1: "Tuberculosis"}  

# Model containers
lung_model = None
brain_model = None

# ---------- Utility Functions ----------

def preprocess_image(img: Image.Image):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def load_and_prepare_image(file: UploadFile):
    """Reads image file and returns preprocessed image array."""
    contents = file.file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    return preprocess_image(img)

def make_prediction(model, img_array, class_labels, binary=False):
    prediction = model.predict(img_array)
    if binary:
        predicted_class = int(prediction[0][0] > 0.5)
        confidence = float(prediction[0][0])
    else:
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

    label = class_labels[predicted_class]
    return label, confidence, prediction.tolist()

# ---------- Model Loaders ----------

def load_lung_model():
    global lung_model
    path = kagglehub.model_download("zeyadabdo/lung-cancer-resnet/keras/v1")
    lung_model_path = os.path.join(path, "lung-cancer-resnet-model.h5")
    lung_model = load_model(lung_model_path, compile=False) 

def load_brain_model():
    global brain_model
    path = kagglehub.model_download("khalednabawi/brain-tumor-cnn/keras/v1")
    brain_model_path = os.path.join(path, "cnn_brain_tumor_model.h5")
    brain_model = load_model(brain_model_path, compile=False)
    
    
@app.on_event("startup")
async def load_models():
    try:
        load_brain_model()
        print("✅ Brain tumor model loaded.")
    except Exception as e:
        print(f"❌ Failed to load Brain tumor model: {e}")
    try:
        load_lung_model()
        print("✅ lung cancer model loaded.")
    except Exception as e:
        print(f"❌ Failed to load lung cancer model: {e}")


# ---------- Routes ----------

@app.get("/")
def root():
    return {"message": " Multi-Disease Detection API is running!"}

@app.post("/lung-cancer")
async def predict_lung(file: UploadFile = File(...)):
    try:
        img_array = load_and_prepare_image(file)
        label, confidence, raw = make_prediction(lung_model, img_array, LUNG_CLASSES)

        return {
            "success": True,
            "prediction": label,
            "confidence": round(confidence, 4),
            "raw": raw
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/brain-tumor")
async def predict_brain(file: UploadFile = File(...)):
    try:
        img_array = load_and_prepare_image(file)
        label, confidence, raw = make_prediction(brain_model, img_array, BRAIN_CLASSES, binary=True)

        return {
            "success": True,
            "prediction": label,
            "confidence": round(confidence, 4),
            "raw": raw
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
