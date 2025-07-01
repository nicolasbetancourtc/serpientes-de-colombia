from kedro.io import AbstractDataset
import os
import torch
import fsspec
from kedro.io.core import get_filepath_str, get_protocol_and_path


class PytorchModelDataset(AbstractDataset):
    def __init__(self, 
                 file_path: str,
                 map_location: str = "cpu"

                 ):
        super().__init__()
        self.file_path = file_path
        self.map_location = map_location
        
        protocol, path = get_protocol_and_path(file_path)
        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self):
        with self._fs.open(self.file_path, 'rb') as f:
            data = torch.load(f, map_location=self.map_location)
            return data
    def _save(self, data):
        with self._fs.open(self.file_path, 'wb') as f:
            torch.save(data, f)
    def _describe(self):
        return {"path": self.file_path}