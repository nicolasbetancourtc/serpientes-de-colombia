from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    global_label_encoder
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(  # Log
                func=global_label_encoder,
                inputs=['train_set'],
                outputs="label_encoder",
                name="global_label_encoder_node",
            )
            
        ]
    )