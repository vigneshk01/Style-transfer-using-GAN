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
import imageio
import numpy as np
from numpy import asarray
from skimage.io import imread
import logging

import matplotlib.pyplot as plt
from skimage.transform import resize

# Import custom InstanceNormalization
from InstanceNormalization import InstanceNormalization

log = logging.getLogger('api')
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)

app = FastAPI()

# Static file and template configuration
templates = Jinja2Templates(directory="./")  # Fixed path to templates folder
app.mount("/static", StaticFiles(directory="./"), name="static")  # Static folder mount

# Path where uploaded and generated images will be stored
UPLOAD_DIR = Path("./static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

##############################################################################################################################
#Defining Downsample and upsample
def downsample(filters, size, apply_norm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()

    # Add Conv2d layer
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    # Add Normalization layer
    if apply_norm:
        result.add(InstanceNormalization())

    # Add Leaky Relu Activation
    # result.add(tf.keras.layers.LeakyReLU())
    result.add(tf.keras.layers.LeakyReLU(alpha = 0.2))
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()

    # Add Transposed Conv2d layer
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    # Add Normalization Layer
    result.add(InstanceNormalization())

    # Conditionally add Dropout layer
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    # Add Relu Activation Layer
    result.add(tf.keras.layers.ReLU())
    return result

#################################################################################################################################################

IMG_HEIGHT, IMG_WIDTH = 256, 256

# Unet Generator is a combination of Convolution + Transposed Convolution Layers
def unet_generator():

    down_stack = [
        downsample(64, 4, True),
        downsample(128, 4, True),
        downsample(256, 4, True),
    ]
    up_stack = [
        upsample(256, 4, False),
        upsample(128, 4, False),
        upsample(64, 4, True)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(1, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh') # (bs, 32, 32, 1)
    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 1])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

#########################################################################################################################################################

# Generator_G = unet_generator()
# Generator_F = unet_generator()

# # Load pre-trained CycleGAN models for T1 -> T2 and T2 -> T1 conversion using Keras
# Generator_G.load_weights("./model/test_G.weights.h5")  # Path to T1 -> T2 generator model
# Generator_F.load_weights("./model/test_F.weights.h5")  # Path to T2 -> T1 generator model

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
    # Convert image to grayscale since model expects 1 channel
    image = image.convert('L')  # 'L' mode = grayscale

    # Resize
    image = image.resize((256, 256))

    # Convert to NumPy array and normalize
    image_array = np.array(image)
    normalized_image = normalize_image(image_array)

    # Expand dims: [256, 256] → [256, 256, 1] → [1, 256, 256, 1]
    normalized_image = np.expand_dims(normalized_image, axis=-1)
    normalized_image = np.expand_dims(normalized_image, axis=0)

    # Predict
    prediction = model.predict(normalized_image)

    # Post-process
    prediction = tf.squeeze(prediction, axis=0)  # Remove batch dimension
    prediction = tf.squeeze(prediction, axis=-1)  # Remove channel dimension

    # Rescale from [-1, 1] → [0, 255]
    prediction = ((prediction + 1.0) * 127.5).numpy()
    prediction = np.clip(prediction, 0, 255).astype(np.uint8)

    # Convert back to image
    result_image = Image.fromarray(prediction)

    return result_image


@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-file")
async def upload_and_predict(file: UploadFile = File(...), conversion_type: str = "T1_to_T2"):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPEG or PNG image.")
    
    # Save the uploaded file
    sanitized_filename = sanitize_filename(file.filename)
    file_path = UPLOAD_DIR / sanitized_filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Open the image file
    uploaded_image = Image.open(file_path)

    # Apply the appropriate transformation
    if conversion_type == "T1_to_T2":
        # transformed_image = transform_image(uploaded_image, Generator_G)
        transformed_image = uploaded_image
    elif conversion_type == "T2_to_T1":
        # transformed_image = transform_image(uploaded_image, Generator_F)
        transformed_image = uploaded_image
    else:
        raise HTTPException(status_code=400, detail="Invalid conversion type.")

    # Save the transformed image
    transformed_file_path = UPLOAD_DIR / f"transformed_{sanitized_filename}"
    transformed_image.save(transformed_file_path)

    return {"filename": transformed_file_path.name, "file_size": os.path.getsize(transformed_file_path)}

@app.post("/batch-upload")
async def batch_upload(images: List[UploadFile], conversion_type: str = "T1_to_T2"):
    result_files = []

    # Loop through all the uploaded files
    for file in images:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPEG or PNG image.")
        
        # Save the uploaded file
        sanitized_filename = sanitize_filename(file.filename)
        file_path = UPLOAD_DIR / sanitized_filename
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Open the image file
        uploaded_image = Image.open(file_path)

        # Apply the appropriate transformation
        if conversion_type == "T1_to_T2":
            transformed_image = transform_image(uploaded_image, Generator_G)
            #transformed_image = uploaded_image
        elif conversion_type == "T2_to_T1":
            transformed_image = transform_image(uploaded_image, Generator_F)
            #transformed_image = uploaded_image
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
