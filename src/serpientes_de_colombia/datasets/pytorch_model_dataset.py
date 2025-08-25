from kedro.io import AbstractDataset
import os
import torch
import fsspec
from torchvision.models import get_model
import torch.nn as nn
from kedro.io.core import get_filepath_str, get_protocol_and_path


class PytorchModelDataset(AbstractDataset):
    def __init__(self, 
                 file_path: str,
                 backbone_model: str,
                

                 ):
        super().__init__()
        self.file_path = file_path
        self.backbone_model = backbone_model
        # get protocol and path from file_path
        
        protocol, path = get_protocol_and_path(file_path)
        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol)
    def _save(self, model):
        payload = {
            "state_dict": model.state_dict(),
            "num_classes": model.fc.out_features,
        }
        with self._fs.open(self.file_path, 'wb') as f:
            torch.save(payload, f)
    def _load(self):
        model = get_model(self.backbone_model)
        
        with self._fs.open(self.file_path, 'rb') as f:
            payload = torch.load(f)

        num_classes = payload["num_classes"]
        state_dict = payload["state_dict"]

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(state_dict)

        return model
    def _describe(self):
        return {"Path": self.file_path, "Backbone model": self.backbone_model}