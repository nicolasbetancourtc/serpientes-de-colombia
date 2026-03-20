from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    cf_matrix,
    loss_history_plot,
    get_sample_predictions,
    save_sample_predictions
)

def create_pipeline(**kwargs):
    return Pipeline(
       
        [  node(  # Log
                func=cf_matrix,
                inputs=["predictions@pandas", 'label_encoder'],
                outputs="confusion_matrix@matplotlib",
                name="confusion_matrix_node",
            ),
            
            node(  # Log
                    func=loss_history_plot,
                    inputs=f"loss_history@pandas",
                    outputs="loss_history_plot@matplotlib",
                    name=f"loss_history_plot_node",
                ),
            
            node(  # Log
                    func=get_sample_predictions,
                    inputs=["predictions@pandas", "params:sample_size"],
                    outputs="sample_predictions@pandas",
                        name=f"get_sample_predictions_node",
                ),
            node(  # Log
                    func=save_sample_predictions,
                    inputs="sample_predictions@pandas",
                    outputs="sample_predictions_folder",
                        name=f"save_sample_predictions_node",
                ),

        
            
        ]
    )