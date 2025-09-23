"""Project pipelines."""
from pathlib import Path

from kedro.pipeline import Pipeline
from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings
from kedro.framework.project import find_pipelines

from serpientes_de_colombia.pipelines.data_engineering.raw import pipeline as raw
from serpientes_de_colombia.pipelines.data_engineering.intermediate import pipeline as intermediate
from serpientes_de_colombia.pipelines.data_engineering.primary import pipeline as primary

from serpientes_de_colombia.pipelines.data_science.model_input import pipeline as model_input
from serpientes_de_colombia.pipelines.data_science.model import pipeline as model
from serpientes_de_colombia.pipelines.data_science.model_output import pipeline as model_output
from serpientes_de_colombia.pipelines.data_science.reporting import pipeline as reporting




def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    #Data Engineering Pipelines
    raw_pipeline = raw.create_pipeline()
    intermediate_pipeline = intermediate.create_pipeline()
    primary_pipeline = primary.create_pipeline()
    model_input_pipeline = model_input.create_pipeline()

    #Data Science Pipelines
    project_root = Path(__file__).resolve().parents[2]  # up from src/
    conf_loader = OmegaConfigLoader(conf_source=str(project_root / settings.CONF_SOURCE), base_env="base")

    params = conf_loader["parameters"]
    model_pipeline = model.create_pipeline(backbone_models=params["backbone_models"])
    model_output_pipeline = model_output.create_pipeline(backbone_models=params["backbone_models"])
    reporting_pipeline = reporting.create_pipeline(backbone_models=params["backbone_models"])

    pipelines = {
        "__default__": intermediate_pipeline+primary_pipeline+model_input_pipeline+model_pipeline+model_output_pipeline+raw_pipeline+reporting_pipeline,
        "raw": raw_pipeline,
        "intermediate": intermediate_pipeline,
        "primary": primary_pipeline,    
        "models": model_pipeline,
        "model_input": model_input_pipeline,
        "model_output": model_output_pipeline,
        "reporting": reporting_pipeline,
    }
    return pipelines
