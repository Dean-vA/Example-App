import argparse
import os

import mlflow
from load_data import (load_and_preprocess_data,
                       load_and_preprocess_data_from_uri)
from model import load_model
import logging

logging.basicConfig(level=logging.INFO)

def evaluate_model(model, use_uri=False, uri=None, output_dir="outputs"):
    logging.info("Evaluating model")
    logging.info(f"Loading test data")
    if not use_uri:
        # Load and preprocess data
        (_, _), (test_images, test_labels) = load_and_preprocess_data()
    else:
        test_images, test_labels = load_and_preprocess_data_from_uri(uri)
    logging.info(f"Test data loaded")

    # Load the model
    logging.info(f"Loading model from {model}")
    model = load_model(model)
    
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    # Log model performance to MLflow
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # check the output directory and file exists and if not create them
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the accuracy to a file in the output directory
    with open(output_dir + "/accuracy.txt", "w") as f:
        f.write(str(test_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a TensorFlow model.")
    parser.add_argument("--model_path", type=str, help="Path to the saved model.")
    parser.add_argument(
        "--use_uri",
        action="store_true",
        help="Use the MNIST dataset from the Keras API",
    )
    parser.add_argument(
        "--test_data_dir", type=str, help="URI of the MNIST dataset from the Keras API"
    )
    parser.add_argument(
        "--accuracy_path", type=str, default="outputs", help="Path to the accuracy txt"
    )
    args = parser.parse_args()

    evaluate_model(
        args.model_path, args.use_uri, args.test_data_dir, args.accuracy_path
    )
