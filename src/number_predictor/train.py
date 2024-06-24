import argparse
import os
# set tensorflow memory growth to true
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from load_data import (load_and_preprocess_data,
                       load_and_preprocess_data_from_uri)
from model import create_model
import logging

logging.basicConfig(level=logging.INFO)

def train(use_uri, uri=None, model_path=None, test_train_ratio=0.1) -> None:
    """
    This function takes in three arguments. The use_uri flag indicates whether to use the built-in data
    or to load it from a specified URI. The uri is a string that points to the data's location, and model_path
    is a string specifying where to save the trained model.

    The function loads and preprocesses the MNIST data, creates and trains a model, evaluates its performance,
    and saves the trained model.

    Args:
    use_uri : bool
        Flag that determines whether to use built-in data or load from a uri
    uri : str, optional
        Path to load the data from, by default None
    model_path : str, optional
        Path where to save the trained model, by default None
    """
    # Check if gpu is available
    if tf.test.is_gpu_available():
        print("The GPU will be used for training.")
    else:
        print("The CPU will be used for training.")

    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"Gpus: {physical_devices}")
    if len(physical_devices) > 0:
        print("Setting memory growth for GPU")
    else:
        print("No GPU found :(")
        
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        print("Could not set memory growth for GPU")
        pass

    # Start MLflow tracking
    mlflow.start_run()
    # mlflow.tensorflow.autolog()

    if not use_uri:
        # Load and preprocess data
        (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_data()
    else:
        train_images, train_labels = load_and_preprocess_data_from_uri(uri)

    # Create the model
    model = create_model()

    # Train the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    history = model.fit(
        train_images, train_labels, epochs=10, validation_split=test_train_ratio
    )

    # Log the model summary to MLflow
    mlflow.log_text("model_summary.txt", str(model.summary()))

    # Log the model metrics to MLflow
    mlflow.log_metric("train_loss", history.history["loss"][-1])
    mlflow.log_metric("train_accuracy", history.history["accuracy"][-1])
    mlflow.log_metric("val_loss", history.history["val_loss"][-1])
    mlflow.log_metric("val_accuracy", history.history["val_accuracy"][-1])

    # plot the training and validation loss and accuracies for each epoch using matplotlib and log the image to MLflow
    fig = plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    mlflow.log_figure(fig, "metrics.png")

    # check if the model directory exists
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Save the trained model
    logging.info(f"Saving model to {model_path}/model.keras")
    print(f"Saving model to {model_path}/model.keras")
    model.save(f'{model_path}/model.keras')
    # Show the files in the model directory
    logging.info(f"Files in {model_path}")
    for root, dirs, files in os.walk(model_path):
        logging.info(f"Root: {root}")
        logging.info(f"Dirs: {dirs}")
        logging.info(f"Files: {files}")

    # Log the model to MLflow
    # mlflow.tensorflow.log_model(model, "model")

    # End the MLflow run
    mlflow.end_run()


if __name__ == "__main__":
    # define arguments using argparse
    parser = argparse.ArgumentParser(
        description="Train a model to recognize handwritten digits from the MNIST dataset."
    )
    parser.add_argument(
        "--use-uri",
        action="store_true",
        help="Use the MNIST dataset from the Keras API",
    )
    parser.add_argument(
        "--data-path", type=str, default="data", help="Path to the MNIST dataset"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/mnist_model",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--test-train-ratio",
        type=float,
        default=0.1,
        help="Ratio of test to train data",
    )
    args = parser.parse_args()

    train(args.use_uri, args.data_path, args.model_path, args.test_train_ratio)
