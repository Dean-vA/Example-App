import argparse
import os
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

def register_model_if_accuracy_above_threshold(model_path, accuracy_folder, threshold=0.5):
    print(f"Registering model if accuracy is above {threshold}.")
    print(f"Model path: {model_path}")
    print(f"Accuracy file: {accuracy_folder}")
    # Get the accuracy file
    accuracy_file = os.path.join(accuracy_folder, "accuracy.txt")
    # Load accuracy from file
    with open(accuracy_file, 'r') as f:
        print(f"Reading accuracy from {accuracy_file}")
        accuracy = float(f.read().strip())

    print(f"Model accuracy: {accuracy}")
    
    # Only register model if accuracy is above threshold
    if accuracy > threshold:
        print("Model accuracy is above threshold, registering model.")
        
        # Define your Azure ML settings
        subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
        resource_group = "dean-sandbox"
        workspace_name = "adsaimlsandbox"
        tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
        client_id = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
        client_secret = "aTw8Q~wmEvkNZcjcVOu.l1PL8KZ_sF~VJ3zvZc2b"

        credential = ClientSecretCredential(tenant_id, client_id, client_secret)

        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

        model = Model(
            path=model_path,
            type=AssetTypes.CUSTOM_MODEL,
            name="example_2",
            description="Model created from pipeline",
        )

        # Register the model
        model = ml_client.models.create_or_update(model)
        #print(f"Model {model.name} registered.")
        print(f"Model registered.")
    else:
        print("Model accuracy is not above threshold, not registering model.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Register a TensorFlow model if accuracy is above threshold.')
    parser.add_argument('--model', type=str, help='Path to the saved model.')
    parser.add_argument('--accuracy', type=str, help='Path to the file containing model accuracy.')
    args = parser.parse_args()

    register_model_if_accuracy_above_threshold(args.model, args.accuracy)
