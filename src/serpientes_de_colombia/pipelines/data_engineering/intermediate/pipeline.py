from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    
    filter_failed_downloads
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            
            node(  # Log
                func=filter_failed_downloads,
                inputs=[ "serpientes_de_colombia_images", "unfiltered_image_urls@pandas"],
                outputs="image_urls@pandas",
                name="filter_failed_downloads_node",
            ),
            
            
        ]
    )