from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
fxd_feature_extractor    
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=fxd_feature_extractor,
                inputs=["train_image_dataset" , 
                        "test_image_dataset" ,
                        "params:dense_params" ,
                        'label_encoder',
                        'device',
                        f"params:backbone_models.{parameter_name}"
                        ],
                outputs=f"{backbone_model}_model",
                name=f"fxd_feature_extractor_{backbone_model}_node",
            )
            
            
            for parameter_name, backbone_model in  kwargs["backbone_models"].items()
            
        ]
    )