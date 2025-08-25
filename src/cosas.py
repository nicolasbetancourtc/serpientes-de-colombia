from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

print(settings)
print(settings.CONF_SOURCE)
#conf_loader = OmegaConfigLoader(conf_source=settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source="conf", base_env="base")
params = conf_loader["parameters"]
print("PARAMS:", params['backbone_models'])
