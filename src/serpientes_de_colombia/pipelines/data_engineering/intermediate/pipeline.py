from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    list_train_files
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(  # Log
                func=list_train_files,
                inputs=["raw_image_dataset"],
                outputs="image_dataset",
                name="list_train_files_node",
            ),
            
            
        ]
    )