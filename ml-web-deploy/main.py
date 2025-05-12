# This is a FastAPI application that serves a machine learning model for image conversion.
# It allows users to upload an image, processes it, and returns the converted image.
# The application uses TensorFlow and Keras for model loading and prediction.
# The application also includes a simple HTML interface for file upload and displays the result.

# Import necessary libraries
import os
import re
import matplotlib.image
import tensorflow as tf

from numpy import asarray
from tensorflow import keras
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile
from fastapi.templating import Jinja2Templates
from InstanceNormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array, load_img

# Initialize FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="./")  # Setup templates

# Load the model with custom layers
T1ToT2ImageConverter = keras.models.load_model(
    "../edits/test_g_new.keras", custom_objects={"InstanceNormalization": InstanceNormalization})

# Set up directories
UPLOAD_DIR = "./static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="./"), name="static")


# Function to cleanup the file name by removing special characters
def sanitize_filename(filename: str) -> str:
    # Remove spaces and any special characters (non-alphanumeric and non-period or underscore)
    sanitized_filename = re.sub(r"[^A-Za-z0-9_.]", "", filename)
    return sanitized_filename


@app.get("/")
# Render the HTML template for the main page
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
# Handle file upload and prediction
async def upload_and_predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        contents = await file.read()
        file_size = len(contents)
        with open(file_path, "wb") as f:
            f.write(contents)

        # Predict using the model
        prediction = test_image(file_path)
        file_name = sanitize_filename(file.filename)

        # Save result
        output_path = os.path.join(UPLOAD_DIR, f"converted_{file_name}")
        matplotlib.image.imsave(output_path, prediction[0], cmap="gray")

        return JSONResponse(
            content={
                "filename": file_name,
                "output_path": f"/static/static/uploads/converted_{file_name}",
                "file_size": file_size,
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


def test_image(img):
    # preprocess the data, predict amd plot the test images
    test_image = load_img(img, target_size=(256, 256))
    test_image = img_to_array(test_image)
    test_image = asarray(test_image)
    test_image = tf.image.rgb_to_grayscale(test_image)
    test_image = tf.expand_dims(test_image, axis=0)
    test = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(test_image))

    for p in test.take(1):
        return T1ToT2ImageConverter(test_image)
