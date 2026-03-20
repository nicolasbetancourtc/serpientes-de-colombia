from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    train_validation_test_split,
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
           
            node(  # Log
                func=train_validation_test_split,
                inputs=['image_urls@pandas',     
                        'params:train_validation_size',     
                        'params:validation_size',
                        ],
                outputs=['train_set', 'validation_set', 'test_set'],
                name="train_validation_test_split_node",
            ),
            
        ]
    )