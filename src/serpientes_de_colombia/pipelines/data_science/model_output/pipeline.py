from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    get_predictions
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(  # Log
                func=get_predictions,
                inputs=['test_image_dataset','resnet18_model', 'label_encoder', "params:dense_params", 'device'],
                outputs="dense_predictions",
                name="get_predictions_node",
            ),
            
            
        ]
    )