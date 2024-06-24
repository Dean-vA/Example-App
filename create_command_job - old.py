# This is old please see the notebook for the new code

from azure.ai.ml import Input, MLClient, command
from azure.identity import InteractiveBrowserCredential

credential = InteractiveBrowserCredential()

subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "dean-sandbox"
workspace_name = "adsaimlsandbox"

ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

env = ml_client.environments.get("aml-keras-mnist", 11)
compute_target = "cloud"
# This is the path to the data folder in the datastore, click on the data asset in the Azure ML portal to get the path
path = "azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/dean-sandbox/workspaces/adsaimlsandbox/datastores/datastore/paths/mnist/train"

job = command(
    inputs=dict(
        data=Input(type="uri_folder", description="Data asset URI", path=path),
        test_train_ratio=0.2,
    ),
    code="./src/number_predictor",  # location of source code
    command="python train.py --use-uri --data-path ${{inputs.data}} --test-train-ratio ${{inputs.test_train_ratio}}",
    environment=env,
    compute=compute_target,  # delete this line to use serverless compute
    display_name="number prediction",
)

ml_client.create_or_update(job)
