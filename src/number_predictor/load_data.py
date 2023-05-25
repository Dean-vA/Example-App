import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from PIL import Image
import os

def load_and_preprocess_data():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocess the data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)

def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from the specified path, convert it to grayscale, resize it to 28x28 pixels, and normalize its pixel values.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        np.ndarray: Preprocessed image array.
    """

    if os.path.isfile(image_path):
        with Image.open(image_path) as img:
            img = img.convert('L')  # Convert to grayscale
            img = img.resize((28, 28), Image.ANTIALIAS)  # Resize to 28x28 pixels
            image = np.array(img) / 255.0  # Normalize pixel values
            return image
    else:
        print(f"No image file found at {image_path}")
        return None