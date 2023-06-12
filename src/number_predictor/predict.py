import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from load_data import load_and_preprocess_image
from model import load_model

def predict_digit(model: tf.keras.Model, image: np.ndarray) -> int:
    """
    Predict the digit represented by the input image using the provided model.

    Args:
        model (tf.keras.Model): Trained model to use for prediction.
        image (np.ndarray): Preprocessed image array.

    Returns:
        int: Predicted digit.
    """
    # Ensure the image has the right dimensions for the model (batch, height, width, channels)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    # Make predictions
    predictions = model.predict(image)
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions)

    return predicted_digit, confidence

def main(args):
    """
    Main function.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    path = args.image_path
    # Load the saved model
    model = load_model('../../models/mnist_model')

    # Load and preprocess the input image
    image = load_and_preprocess_image(args.image_path)

    # Predict the digit
    predicted_digit, confidence = predict_digit(model, image)

    print(f'Predicted digit: {predicted_digit}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict a digit from an image using a trained MNIST model.')
    parser.add_argument('image_path', help='Path to the input image file.')
    args = parser.parse_args()
    main(args)
