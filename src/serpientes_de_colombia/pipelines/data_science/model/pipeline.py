from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
dense_training    
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(  # Log
                func=dense_training,
                inputs=["train_image_dataset", 'label_encoder','params:dense_params'],
                outputs="dense_model",
                name="dense_training_node",
            ),
            
        ]
    )