import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from PIL import Image
import os
import io
from typing import Tuple

def load_and_preprocess_data_from_uri(uri: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses image data from a given URI.

    The function assumes a directory structure where each subdirectory's name is 
    an integer that corresponds to the label of all images within that directory.
    Images are assumed to be .png files.

    Parameters
    ----------
    uri : str
        URI of the directory containing subdirectories of images.

    Returns
    -------
    tuple of np.ndarray
        Returns a tuple of numpy arrays, the first array contains the images,
        the second array contains the corresponding labels.
    """
    
    print(f'Loading data from {uri}...')
    print('Found the following folders in the directory:', os.listdir(uri))

    image_list = []
    label_list = []

    # Iterate over all folders in base_path
    for folder_name in os.listdir(uri):
        folder_path = os.path.join(uri, folder_name)
        
        if os.path.isdir(folder_path):
            # Ensure the folder name can be converted to an integer (i.e., 0-9 or any digit)
            try:
                label = int(folder_name)
            except ValueError:
                print(f'Non-integer folder name {folder_name} encountered. Skipping this folder.')
                continue
            
            # Iterate over all files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith('.jpg'):  # Assuming images are .png. Change this if needed
                    image_path = os.path.join(folder_path, filename)
                    image = Image.open(image_path)
                    image_np = np.array(image)  # Convert to numpy array
                    image_list.append(image_np)
                    label_list.append(label)

    # Convert the lists to numpy arrays and normalize images to be between 0 and 1
    train_images = np.array(image_list) / 255.0
    train_labels = np.array(label_list)

    print(f'Loaded {len(train_images)} images and {len(train_labels)} labels.')

    return (train_images, train_labels)

def load_and_preprocess_data():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocess the data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return (train_images, train_labels)

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
    
def load_and_preprocess_image_file(image_file: io.BytesIO) -> np.ndarray:
    """
    Load an image file and preprocess it for prediction.

    Args:
        image_file (io.BytesIO): Input image file.

    Returns:
        np.ndarray: Preprocessed image.
    """
    # Load the image file
    image = Image.open(image_file)

    # Convert the image to grayscale
    image = image.convert('L')

    # Resize the image to match the input shape that the model expects
    image = image.resize((28, 28))

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Normalize the image array
    image_array = image_array / 255.0

    return image_array