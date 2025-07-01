from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    list_train_files,
    temp
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(  # Log
                func=list_train_files,
                inputs=["raw_image_dataset"],
                outputs="train_data",
                name="list_train_files_node",
            ),
            node(  # Log
                func=temp,
                inputs=["train_data"],
                outputs="aaa",
                name="test_node",
            ),
            
        ]
    )