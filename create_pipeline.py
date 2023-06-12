from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.ai.ml import dsl

subscription_id = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
resource_group = "dean-sandbox"
workspace_name = "adsaimlsandbox"
tenant_id = "0a33589b-0036-4fe8-a829-3ed0926af886"
client_id = "a2230f31-0fda-428d-8c5c-ec79e91a49f5"
client_secret = "aTw8Q~wmEvkNZcjcVOu.l1PL8KZ_sF~VJ3zvZc2b"

service_principal = ServicePrincipalAuthentication(
    tenant_id=tenant_id,
    service_principal_id=client_id,
    service_principal_password=client_secret,
)

workspace = Workspace(subscription_id=subscription_id,
                      resource_group=resource_group,
                      workspace_name=workspace_name,
                      auth=service_principal
                      )

credential = ClientSecretCredential(tenant_id, client_id, client_secret)

ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

environments = ml_client.environments.list()
print('Environments:\n')
for environment in environments:
    print(environment.name,':', )
print('-------------------')

# List all available datasets
datasets = ml_client.data.list()
print('Datasets:\n')
for dataset in datasets:
    print(dataset.name,':', dataset.latest_version)
print('-------------------')

# List all available compute targets
compute_targets = ml_client.compute.list()
print('Compute targets:\n')
for compute_target in compute_targets:
    print(compute_target.name,'-', compute_target.type)
print('-------------------')

environment_name = 'aml-keras-mnist'
environment_version = 11
print(f'Using environment {environment_name} version {environment_version}')
compute_target_name = 'cloud'

component_path = "./src/number_predictor/"
env = ml_client.environments.get(environment_name, environment_version)


train_component = command(
                        name="train",
                        display_name="Train model",
                        description="Train model with data from a predefined data asset",
                        inputs={
                            "data": Input(type="uri_folder", description="Data asset URI"),
                        },
                        outputs=dict(model=Output(type="uri_folder", mode="rw_mount")),
                        code=component_path,
                        command="python train.py --use-uri --data-path ${{inputs.data}} --model-path ${{outputs.model}}",
                        environment=env,
                        #compute_target=compute_target.name,
                    )

train_component = ml_client.create_or_update(train_component.component)

evaluate_component = command(
                        name="evaluate",
                        display_name="Evaluate model",
                        description="Evaluate model with data from a predefined data asset",
                        inputs={
                            "data": Input(type="uri_folder", description="Data asset URI"),
                            "model": Input(type="uri_folder", description="Model URI"),
                        },
                        outputs=dict(
                           accuracy=Output(type="uri_folder", description="Model accuracy output")
                        ),
                        code=component_path,
                        command="python evaluate.py --use_uri --test_data_dir ${{inputs.data}} --model_path ${{inputs.model}} --accuracy_path ${{outputs.accuracy}}",
                        environment=env,
                        #compute_target=compute_target.name,
                    )

evaluate_component = ml_client.create_or_update(evaluate_component.component)

register_component = command(
                        name="register",
                        display_name="Register model",
                        description="Register model with data from a predefined data asset",
                        inputs={
                            "model": Input(type="uri_folder", description="Model URI"),
                            "accuracy": Input(type="uri_folder", description="Model accuracy file"),
                        },
                        code=component_path,
                        command="python register.py --model ${{inputs.model}} --accuracy ${{inputs.accuracy}}",
                        environment=env,
                        #compute_target=compute_target.name,
                    )

register_component = ml_client.create_or_update(register_component.component)


#list all components
components = ml_client.components.list()
print('Components:\n')
for component in components:
    print(component.name,':', component.version)
print('-------------------')

@dsl.pipeline(
    name='Example pipeline',
    compute='cloud',#compute_target.name,
    instance_type="defaultinstancetype"
    )
def train_eval_reg_pipeline(
    train_data_asset_uri: str,
    test_data_asset_uri: str,
) -> None:
    
    training_step = train_component(data=train_data_asset_uri)
    evaluation_step = evaluate_component(data=test_data_asset_uri, model=training_step.outputs.model)
    register_step = register_component(model=training_step.outputs.model, accuracy=evaluation_step.outputs.accuracy)

train_digits_ds = Input(path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/dean-sandbox/workspaces/adsaimlsandbox/datastores/datastore/paths/mnist/train")
test_digits_ds = Input(path="azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/resourcegroups/dean-sandbox/workspaces/adsaimlsandbox/datastores/datastore/paths/mnist/test")
# Instantiate the pipeline.
pipeline_instance = train_eval_reg_pipeline(train_data_asset_uri=train_digits_ds, test_data_asset_uri=test_digits_ds)

# Submit the pipeline.
pipeline_run = ml_client.jobs.create_or_update(pipeline_instance)






