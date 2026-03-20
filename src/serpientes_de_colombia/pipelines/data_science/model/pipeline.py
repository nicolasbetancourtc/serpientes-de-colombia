from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
fxd_feature_extractor    
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=fxd_feature_extractor,
                inputs=["train_set" , 
                        "validation_set", 
                        "params:training_params" ,
                        'label_encoder',
                        'device',
                        f"params:backbone_model"
                        ],
                outputs=[f"model", f"loss_history@pandas"],
                name=f"fxd_feature_extractor_node",
            )
            
            
            
            
        ]
    )