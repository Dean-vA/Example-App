from azure.ai.ml import Input, MLClient, command
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Environment

credential = InteractiveBrowserCredential()

subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "buas-y2"
workspace_name = "Staff-Test"

ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

env = ml_client.environments.get("tf-gpu-docker-2", 2)
compute_target = "adsai1"
data_asset = ml_client.data.get("digits_val_2", version="1")

# job = command(
#     inputs=dict(
#         data=Input(type="uri_folder", description="Data asset URI", path=path),
#         test_train_ratio=0.2,
#     ),
#     code="./src/number_predictor",  # location of source code
#     command="python train.py --use-uri --data-path ${{inputs.data}} --test-train-ratio ${{inputs.test_train_ratio}}",
#     environment=env,
#     compute=compute_target,  # delete this line to use serverless compute
#     display_name="number prediction",
# )

# ml_client.create_or_update(job)

# to successfully create a job, customize the parameters below based on your workspace resources
job = command(
        #command='ls "${{inputs.data}}"',
        command='find "${{inputs.data}}"',
        inputs={
            "data": Input(path=data_asset.id,
                type=AssetTypes.MLTABLE,
                mode=InputOutputModes.RO_MOUNT
            )
        },
        environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
        compute=compute_target,
        instance_type="cpu-med"
      )
returned_job = ml_client.jobs.create_or_update(job)


