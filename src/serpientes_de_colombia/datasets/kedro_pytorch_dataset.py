from pathlib import Path
from typing import Any, Dict
from kedro.io import AbstractDataset
from torch.utils.data import Dataset
import fsspec

from kedro.io.core import get_filepath_str, get_protocol_and_path
import pandas as pd
import os
from PIL import Image
class KedroPytorchImageDataset(AbstractDataset, Dataset):
    def __init__(
        self,
        path_to_images: str, #path to the folder containing the subfolders with images labeled by subfolder's name
        path_column: str = "path",
        label_column: str = "label",
        labels_csv: str = "data.csv",
        transform: Any = None,
        target_transform: Any = None,
        ):
            super().__init__()
            self.transform = transform
            self.target_transform_fn = target_transform
            self.path_to_images = path_to_images
            self.labels_csv = labels_csv
            self.path_column = path_column
            self.label_column = label_column
            self.data=None

            protocol, path = get_protocol_and_path(self.labels_csv)
            self._protocol = protocol
            self._fs = fsspec.filesystem(self._protocol)
            

    #############
    #############
    ### pytorch Dataset methods
    #############
    #############
    def __len__(self):
            'Denotes the total number of samples'
            return self.data.shape[0]

    def __getitem__(self, index):
        
            file_name = self.data.loc[index,  self.path_column]
            label = self.data.loc[index, self.label_column]
            path = Path(self.path_to_images) / label / file_name
            path = path.as_posix()  # ensures forward slashes
    
            image = Image.open(path).convert('RGB')  # Open image as RGB

            if self.transform:
                image = self.transform(image)
                return image, self.target_transform_fn(label) if self.target_transform_fn else label, path
    def with_transforms(self, transform=None, target_transform=None) -> "KedroPytorchImageDataSet":
        self.transform = transform
        self.target_transform_fn = target_transform
        return self
    #############
    #############
    ### KEDRO Dataset methods
    #############
    #############
    def _load(self) -> "KedroPytorchImageDataSet":
        self.data=pd.read_csv(self.labels_csv)
        return self
        
    def _save(self, data: pd.DataFrame) -> None:
        self.data=data
        with self._fs.open(self.labels_csv, "w") as f:
            data.to_csv(f, index=False)
    def _describe(self) -> Dict[str, Any]:
        return {'directory': self.path_to_images,
                'num_examples': self.data.shape[0] if self.data is not None else 'No data loaded',
                'num_classes': self.data[self.label_column].nunique() if self.data is not None else 'No data loaded',
                'classes': self.data[self.label_column].unique().tolist() if self.data is not None else 'No data loaded',
                'transform': self.transform,
                'target_transform': self.target_transform_fn
                }
    
