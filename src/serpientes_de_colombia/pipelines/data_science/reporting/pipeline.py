from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    cf_matrix
)

def create_pipeline(**kwargs):
    return Pipeline(
       
        [  node(  # Log
                func=cf_matrix,
                inputs=[f"{backbone_model}_predictions@pandas", 'label_encoder'],
                outputs=f"confusion_matrix_{backbone_model}@matplotlib",
                name=f"confusion_matrix_{backbone_model}_node",
            )
            
            for idx, (parameter_name, backbone_model) in  enumerate(kwargs["backbone_models"].items())

        
            
        ]
    )