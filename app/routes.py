from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from PIL import Image
import threading
import base64
# ModelLoader Singleton (as before)
class SingletonModelLoader:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_path, scaler_path):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SingletonModelLoader, cls).__new__(cls)
                cls._instance._initialize(model_path, scaler_path)
            return cls._instance

    def _initialize(self, model_path, scaler_path):
        self.model = self._load_model(model_path)
        self.scaler = self._load_model(scaler_path)

    @staticmethod
    def _load_model(model_path):
        """Load the trained model or scaler."""
        with open(model_path, 'rb') as model_file:
            return pickle.load(model_file)

    def predict(self, input_data):
        """Make prediction using the loaded model."""
        input_array = np.array(input_data).reshape(1, -1)  # Flatten image
        input_array = self.scaler.transform(input_array)   # Normalize the input
        prediction = self.model.predict(input_array)
        return prediction[0]  # Return the predicted label


# Pydantic model for incoming image data
class ImageInput(BaseModel):
    image: str  # Base64-encoded image string

# Create a FastAPI router
router = APIRouter()
import logging

logging.basicConfig(level=logging.DEBUG)

@router.post("/predict/")
async def predict(image_input: ImageInput):
    try:
        # Log the received Base64 string
        logging.debug(f"Received Base64 string (first 50 chars): {image_input.image[:50]}...")

        # Decode Base64
        base64_str = image_input.image
        img_data = BytesIO(base64.b64decode(base64_str))
        img_data.seek(0)  # Reset buffer position

        # Open and process the image
        img = Image.open(img_data).convert("L")  # Convert to grayscale
        img = img.resize((8, 8))  # Resize for the model
        img_array = np.array(img).flatten()  # Flatten into 1D array

        logging.debug(f"Image processed successfully. Array shape: {img_array.shape}")

        # Load the model and predict
        model_loader = SingletonModelLoader(model_path="model/svm_model.pkl", scaler_path="model/scaler.pkl")
        prediction = model_loader.predict(img_array)
        return {"prediction": int(prediction)}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")
