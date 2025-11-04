from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    get_device,
    get_taxon_metadata,
    get_image_urls,
    download_images,
    filter_failed_downloads
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
            node(  # Log
                func=get_taxon_metadata,
                inputs=['params:families_of_interest','params:genus_of_interest'],
                outputs="taxon_metadata@pandas",
                name="get_taxon_metadata_node",
            ),
             node(  # Log
                func=get_image_urls,
                inputs=["taxon_metadata@pandas",
                        'params:inaturalist_request_params.INAT_URL',
                        'params:inaturalist_request_params.desired_observations',
                        'params:inaturalist_request_params.max_results_per_page',
                        'params:inaturalist_request_params.place_id'
                        ],
                outputs="unfiltered_image_urls@pandas",
                name="get_image_urls_node",
            ),
            node(  # Log
                func=download_images,
                inputs=["unfiltered_image_urls@pandas"],
                outputs="serpientes_de_colombia_images",
                name="download_images_node",
            ),
            node(  # Log
                func=filter_failed_downloads,
                inputs=[ "serpientes_de_colombia_images", "unfiltered_image_urls@pandas"],
                outputs="image_urls@pandas",
                name="filter_failed_downloads_node",
            ),
        ]
    )