from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    get_predictions
)

def create_pipeline(**kwargs):
    return Pipeline(
       
        [  node(  # Log
                func=get_predictions,
                inputs=['test_image_dataset',f"{backbone_model}_model", 'label_encoder', "params:training_params", 'device'],
                outputs=f"{backbone_model}_predictions@pandas",
                name=f"get_{backbone_model}_predictions_node",
            )
            
            for parameter_name, backbone_model in  kwargs["backbone_models"].items()
            
        ]
    )