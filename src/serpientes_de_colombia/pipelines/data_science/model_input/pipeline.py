from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
global_label_encoder,
labels_previous_check   
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(  # Log
                func=global_label_encoder,
                inputs=['train_set'],
                outputs="label_encoder",
                name="global_label_encoder_node",
            ),
            node(  # Log
                func=labels_previous_check,
                inputs=['train_set','test_set','validation_set','label_encoder'],
                outputs="sample_transform",
                name="labels_previous_check_node",
            ),
            
        ]
    ) 