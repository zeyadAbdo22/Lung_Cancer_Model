from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
from PIL import Image
import io
import os
from fastapi.middleware.cors import CORSMiddleware

# Load the trained CNN model
model = load_model(r"./model_cnn2.h5")  

# Class names that the model predicts
CLASS_NAMES = ["Adenocarcinoma", "Benign", "Squamous Cell Carcinoma"]

# Initialize FastAPI app
app = FastAPI()

# Allow frontend apps (like React or Streamlit) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with specific origins (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Image size that the model expects
IMG_SIZE = 224

# ✅ Add a default root route for health check
@app.get("/")
def read_root():
    return {"message": "✅ Lung Cancer Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the uploaded image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # Convert the image to a numpy array and normalize it
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = int(np.argmax(prediction))  # Get index of highest probability
        predicted_label = CLASS_NAMES[predicted_class]  # Map index to label
        confidence = float(np.max(prediction))  # Get the highest probability (confidence)

        # Return result as JSON
        return JSONResponse(content={
            "prediction_raw": prediction.tolist(),
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4)  # Round confidence to 4 decimal places
        })

    except Exception as e:
        # Handle errors gracefully
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the server (Railway-friendly)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
