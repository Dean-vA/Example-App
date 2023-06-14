from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration
import datetime

# Define your Azure ML settings
subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "dean-sandbox"
workspace_name = "adsaimlsandbox"
tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
client_id = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
client_secret = "aTw8Q~wmEvkNZcjcVOu.l1PL8KZ_sF~VJ3zvZc2b"

credential = ClientSecretCredential(tenant_id, client_id, client_secret)

ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Define an endpoint name
# Example way to define a random name
endpoint_name = "mnist-endpt-" + datetime.datetime.now().strftime("%m%d%H%M%f")
print(f"Endpoint name: {endpoint_name}")

print("Creating endpoint...")
# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name = endpoint_name, 
    description="this is a sample endpoint",
    auth_mode="key"
)

endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print(f"Endpoint {endpoint.name} provisioning state: {endpoint.provisioning_state}")

# # List Models
# for model in ml_client.models.list():
#     print(model.name, 'version:', model.version)

# # List Environments
# for env in ml_client.environments.list():
#     print(env.name, 'version:', env.version)

# endpoint = ml_client.online_endpoints.get(name=endpoint_name)

# print(
#     f'Endpoint "{endpoint.name}" with provisioning state "{endpoint.provisioning_state}" is retrieved'
# )

registered_model_name = "example"
latest_model_version = 1
registered_environment_name = "aml-keras-mnist"
latest_environment_version = 11

# picking the model to deploy. Here we use the latest version of our registered model
model = ml_client.models.get(name=registered_model_name, version=latest_model_version)
# picking the environment to deploy. Here we use the latest version of our registered environment
env = ml_client.environments.get(name=registered_environment_name, version=latest_environment_version)

print("Creating deployment...")
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=env,
    code_configuration=CodeConfiguration(
        code="./src/number_predictor", scoring_script="scoring.py"
    ),
    instance_type="Standard_DS1_v2",#"Standard_DS3_v2",#Standard_D2_v2
    instance_count=1,
)

blue_deployment = ml_client.begin_create_or_update(blue_deployment).result()

ml_client.online_endpoints.get(name=endpoint_name)
