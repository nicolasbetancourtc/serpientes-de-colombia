from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    get_device
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(  # Log
                func=get_device,
                inputs=['params:device_preference'],
                outputs="device",
                name="get_device_node",
            ),
            
        ]
    )