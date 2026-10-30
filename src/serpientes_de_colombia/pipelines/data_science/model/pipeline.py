from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
fxd_feature_extraction_from_backbone,
finetune_from_backbone    
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=fxd_feature_extraction_from_backbone,
                inputs=["train_set" , 
                        "validation_set", 
                        "params:training_params" ,
                        'label_encoder',
                        'device',
                        f"params:backbone_model"
                        ],
                outputs=[f"model_fxd_feature", f"loss_history_fxd_feature@pandas"],
                name=f"fxd_feature_extraction_from_backbone_node",
            ),
        #    node(
        #    func=finetune_from_backbone,
        #    inputs=["train_set" , 
        #            "validation_set", 
        #            "params:training_params" ,
        #            'label_encoder',
        #            'device',
        #            f"params:backbone_model"
        #            ],
        #    outputs=[f"model_fine_tune", f"loss_history_fine_tune@pandas"],
        #    name=f"finetune_from_backbone_node",
        #)
            
            
            
            
        ]
    )