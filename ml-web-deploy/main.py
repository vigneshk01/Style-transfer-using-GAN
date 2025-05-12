# This is a FastAPI application that serves a machine learning model for image conversion.
# It allows users to upload an image, processes it, and returns the converted image.
# The application uses TensorFlow and Keras for model loading and prediction.
# The application also includes a simple HTML interface for file upload and displays the result.

# Import necessary libraries
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
from typing import List
import os
from pathlib import Path
import shutil
from PIL import Image
import tensorflow as tf
import numpy as np
from io import BytesIO
import re
import logging

# Import custom InstanceNormalization
from InstanceNormalization import InstanceNormalization

log = logging.getLogger('api')
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)


# init_loggers()
app = FastAPI()

# Static file and template configuration
templates = Jinja2Templates(directory="./")  # Fixed path to templates folder
app.mount("/static", StaticFiles(directory="./"), name="static")  # Static folder mount

# Path where uploaded and generated images will be stored
UPLOAD_DIR = Path("./static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load pre-trained CycleGAN models for T1 -> T2 and T2 -> T1 conversion using Keras
generator_g = tf.keras.models.load_model("./model/test_G.keras", custom_objects={"InstanceNormalization": InstanceNormalization})  # Path to T1 -> T2 generator model
generator_f = tf.keras.models.load_model("./model/test_F.keras", custom_objects={"InstanceNormalization": InstanceNormalization})  # Path to T2 -> T1 generator model

# Function to cleanup the file name by removing special characters
def sanitize_filename(filename: str) -> str:
    # Remove spaces and any special characters (non-alphanumeric and non-period or underscore)
    sanitized_filename = re.sub(r"[^A-Za-z0-9_.]", "", filename)
    return sanitized_filename

# Helper function to normalize the image to [-1, 1] range
def normalize_image(image: np.array) -> np.array:
    # Normalize image from [0, 255] to [-1, 1]
    return (image / 127.5) - 1.0

# Helper function to preprocess and transform an image using the model
def transform_image(image: Image.Image, model) -> Image.Image:
    # Resize the image to (256, 256) as required by the model
    image = image.resize((256, 256))
    
    # Convert the image to a NumPy array
    image_array = np.array(image)
    
    # Normalize the image to [-1, 1]
    normalized_image = normalize_image(image_array)
    
    # Expand dimensions to match model input (batch size of 1)
    normalized_image = np.expand_dims(normalized_image, axis=0)

    # Get model prediction
    prediction = model.predict(normalized_image)
    
    # Post-process the prediction
    prediction = tf.squeeze(prediction, axis=0)
    prediction = tf.clip_by_value(prediction, 0.0, 255.0)
    prediction = tf.cast(prediction, tf.uint8)
    
    # Convert the tensor to NumPy array
    image_np = prediction.numpy().squeeze()
    
    # Convert back to image
    result_image = Image.fromarray(image_np)
    
    return result_image

@app.get("/")
async def main(request: Request):
    log.info('test')
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def upload_and_predict(file: UploadFile = File(...), conversion_type: str = "T1_to_T2"):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPEG or PNG image.")
    
    # Save the uploaded file
    sanitized_filename = sanitize_filename(file.filename)
    
    file_path = UPLOAD_DIR / sanitized_filename
    log.info(file_path)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Open the image file
    uploaded_image = Image.open(file_path)

    # Apply the appropriate transformation
    if conversion_type == "T1_to_T2":
        transformed_image = transform_image(uploaded_image, generator_g)
    elif conversion_type == "T2_to_T1":
        transformed_image = transform_image(uploaded_image, generator_f)
    else:
        raise HTTPException(status_code=400, detail="Invalid conversion type.")

    # Save the transformed image
    transformed_file_path = UPLOAD_DIR / f"transformed_{sanitized_filename}"
    log.info(transformed_file_path)
    transformed_image.save(transformed_file_path)
    # uploaded_image.save(transformed_file_path)

    # return {"filename": transformed_file_path.name, "file_size": os.path.getsize(transformed_file_path)}
    # return FileResponse(transformed_file_path)
    return JSONResponse(content={"filename": transformed_file_path.name, "file_size": os.path.getsize(transformed_file_path)})

@app.post("/batch-upload")
async def batch_upload(images: List[UploadFile], conversion_type: str = "T1_to_T2"):
    result_files = []

    # Loop through all the uploaded files
    for file in images:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPEG or PNG image.")
        
        log.info('test')
        # Save the uploaded file
        sanitized_filename = sanitize_filename(file.filename)
        file_path = UPLOAD_DIR / sanitized_filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Open the image file
        uploaded_image = Image.open(file_path)

        # Apply the appropriate transformation
        if conversion_type == "T1_to_T2":
            transformed_image = transform_image(uploaded_image, generator_g)
        elif conversion_type == "T2_to_T1":
            transformed_image = transform_image(uploaded_image, generator_f)
        else:
            raise HTTPException(status_code=400, detail="Invalid conversion type.")

        # Save the transformed image
        transformed_file_path = UPLOAD_DIR / f"transformed_{sanitized_filename}"
        transformed_image.save(transformed_file_path)

        result_files.append(transformed_file_path.name)

    return {"files": result_files}

@app.get("/static/uploads/{filename}")
async def get_uploaded_file(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path)
