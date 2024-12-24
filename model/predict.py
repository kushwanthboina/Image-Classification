import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image
import base64
from io import BytesIO

# Load the trained model and scaler
model_path = "model/svm_model.pkl"
scaler_path = "model/scaler.pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def predict_image(image: str):
    # Decode the base64 image string
    img_data = BytesIO(base64.b64decode(image))
    img = Image.open(img_data)
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((8, 8))  # Resize to 8x8 (like the digits dataset)
    img_array = np.array(img).flatten()  # Flatten to 1D array

    # Normalize the image
    img_array = scaler.transform(img_array.reshape(1, -1))

    # Make the prediction
    prediction = model.predict(img_array)
    return prediction[0]
