import base64
import io
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image


def init():
    global model

    # Check if gpu is available
    if tf.test.is_gpu_available():
        print("The GPU will be used for scoring.")
    else:
        print("The CPU will be used for scoring.")
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"Gpus: {physical_devices}")
    if len(physical_devices) > 0:
        print("Setting memory growth for GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU found :(")
        print("Could not set memory growth for GPU")

    base_path = os.getenv("AZUREML_MODEL_DIR")
    print(f"base_path: {base_path}")
    # show the files in the model_path directory
    print(f"list files in the model_path directory")
    # list files and dirs in the model_path directory
    list_files(base_path)
    # model_path = os.path.join(base_path, "INPUT_model")
    # model_path = base_path
    # model_path = os.path.join(base_path, 'model.keras') # local
    model_path = os.path.join(base_path, "INPUT_model", 'model.keras') # azure
    # model_path = os.path.join(base_path, 'saved_model.pb')
    print(f"model_path: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")

def run(raw_data):
    # Load the JSON data from the POST request
    print(f"raw_data: {raw_data}")
    data = json.loads(raw_data)
    print(f"data: {data}")
    # Get the base64-encoded image data
    base64_image = data["data"]
    print(f"base64_image: {base64_image}")
    # Decode the base64 string into bytes
    image_bytes = base64.b64decode(base64_image)
    print(f"image_bytes: {image_bytes}")
    # Open the bytes as an image
    image = Image.open(io.BytesIO(image_bytes))
    # Convert the image to grayscale
    image = image.convert("L")
    # Resize the image to 28x28 pixels, the size expected by the model
    image = image.resize((28, 28))
    # Convert the image to a numpy array and normalize pixel values to [0, 1]
    data = np.array(image) / 255.0
    print(f"data shape: {data.shape}, min: {data.min()}, max: {data.max()}")
    # The model expects a 4D tensor of shape (batch_size, height, width, channels),
    # so we add an extra dimension to the start and end of the array
    data = np.expand_dims(data, axis=(0, -1))
    print(f"expanded data shape: {data.shape}, min: {data.min()}, max: {data.max()}")
    # Make prediction
    prediction = model.predict(data)
    print(f"prediction: {prediction}")
    # Get the predicted label
    predicted_label = np.argmax(prediction, axis=1)[0]
    return json.dumps(predicted_label.tolist())


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in files:
            print("{}{}".format(subindent, f))
