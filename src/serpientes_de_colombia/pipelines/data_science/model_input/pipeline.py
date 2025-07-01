from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
train_test_split    
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(  # Log
                func=train_test_split,
                inputs=["image_dataset",'params:train_size'],
                outputs=["train_image_dataset", "test_image_dataset"],
                name="train_test_split_node",
            ),
            
        ]
    )