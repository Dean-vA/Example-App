import os

import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)


def create_model():
    # Create the model architecture
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    return model


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a saved TensorFlow model from the specified path.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        tf.keras.Model: Loaded model.
    """
    # Check if the model file exists at the specified path and load it if it does.
    logging.info(f"Loading model from {model_path}")
    # walk the directory to see the contents and log them
    for root, dirs, files in os.walk(model_path):
        logging.info(f"Root: {root}")
        logging.info(f"Dirs: {dirs}")
        logging.info(f"Files: {files}")

    if os.path.exists(model_path):
        logging.info("loading model")
        model = tf.keras.models.load_model(f'{model_path}/model.keras')
        logging.info("model loaded")
        return model
    else:
        print(f"No model file found at {model_path}")
        return None
