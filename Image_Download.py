from sklearn import datasets
from PIL import Image
import numpy as np

# Load Digits dataset from sklearn
digits = datasets.load_digits()

# Get the first sample image (8x8)
sample_image = digits.images[1]

# Convert the float array to an integer array (scale pixel values to 0-255)
scaled_image = (sample_image / sample_image.max() * 255).astype(np.uint8)

# Convert the scaled numpy array to a PIL Image
img = Image.fromarray(scaled_image, mode='L')  # Use 'L' for grayscale images

# Save the image to a file (e.g., 'sample_image.png')
img.save("sample_image1.png")

print("Image saved as 'sample_image1.png'")
from PIL import Image
import base64
from io import BytesIO

# Open the downloaded image
img = Image.open("sample_image1.png").convert("L")  # Convert to grayscale
img = img.resize((8, 8))  # Resize to 8x8 pixels

# Encode to Base64
buffered = BytesIO()
img.save(buffered, format="PNG")
img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

print("Base64 Encoded Image:")
print(img_base64)

