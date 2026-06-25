from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    get_predictions
)

def create_pipeline(**kwargs):
    return Pipeline(
       
        [  node(  # Log
                func=get_predictions,
                inputs=['test_set',"model", 'label_encoder', "params:training_params", 'device'],
                outputs=f"predictions@pandas",
                name=f"get_predictions_node",
            ),
            node(  # Log
                func=get_predictions,
                inputs=['train_set',"model", 'label_encoder', "params:training_params", 'device'],
                outputs=f"predictions_train@pandas",
                name=f"get_predictions_train_node",
            )
         
            
        ]
    )