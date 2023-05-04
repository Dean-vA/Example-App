import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from data import load_and_preprocess_data

def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a saved TensorFlow model from the specified path.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        tf.keras.Model: Loaded model.
    """
    return tf.keras.models.load_model(model_path)

def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from the specified path, convert it to grayscale, resize it to 28x28 pixels, and normalize its pixel values.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        np.ndarray: Preprocessed image array.
    """
    with Image.open(image_path) as img:
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((28, 28), Image.ANTIALIAS)  # Resize to 28x28 pixels
        image = np.array(img) / 255.0  # Normalize pixel values
        return image

def predict_digit(model: tf.keras.Model, image: np.ndarray) -> int:
    """
    Predict the digit represented by the input image using the provided model.

    Args:
        model (tf.keras.Model): Trained model to use for prediction.
        image (np.ndarray): Preprocessed image array.

    Returns:
        int: Predicted digit.
    """
    # Ensure the image has the right dimensions for the model
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    # Make predictions
    predictions = model.predict(image)
    predicted_digit = np.argmax(predictions)

    return predicted_digit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict a digit from an image using a trained MNIST model.')
    parser.add_argument('image_path', help='Path to the input image file.')
    args = parser.parse_args()

    # Load the saved model
    model = load_model('../models/mnist_model')

    # Load and preprocess the input image
    image = load_and_preprocess_image(args.image_path)

    # Predict the digit
    predicted_digit = predict_digit(model, image)

    print(f'Predicted digit: {predicted_digit}')
