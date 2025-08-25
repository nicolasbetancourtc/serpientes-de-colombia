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
        path: str, #path to the folder containing the subfolders with images labeled by subfolder's name
        path_column: str = "path",
        label_column: str = "label",
        data_file_name: str = "data.csv",
        transform: Any = None,
        target_transform: Any = None,
        ):
            super().__init__()
            self.transform = transform
            self.target_transform_fn = target_transform
            self.dir_path = path
            self.data_file_name = data_file_name
            self._data_path =os.path.join(self.dir_path, self.data_file_name)
            self.path_column = path_column
            self.label_column = label_column
            self.data=None

            protocol, path = get_protocol_and_path(os.path.join(self.dir_path, self.data_file_name))
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
        
            path = self.data.loc[index, 'path']
            label = self.data.loc[index, 'label']
            

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
        self.data=pd.read_csv(os.path.join(self.dir_path, self.data_file_name))
        return self
        
    def _save(self, data: pd.DataFrame) -> None:
        self.data=data
        with self._fs.open(self._data_path, "w") as f:
            data.to_csv(f, index=False)
    def _describe(self) -> Dict[str, Any]:
        return {'directory': self.dir_path,
                'num_examples': self.data.shape[0] if self.data is not None else 'No data loaded',
                'num_classes': self.data[self.label_column].nunique() if self.data is not None else 'No data loaded',
                'classes': self.data[self.label_column].unique().tolist() if self.data is not None else 'No data loaded',
                'transform': self.transform,
                'target_transform': self.target_transform_fn
                }
    
class OLDKedroPytorchImageDataset(AbstractDataset, Dataset):
   
    def __init__(
        self,
        path: str, #path to the folder containing the subfolders with images labeled by subfolder's name
        path_column: str = "path",
        label_column: str = "label",
        data_file_name: str = "data.csv",
        transform: Any = None,
        target_transform: Any = None,

    ):
        super().__init__()
        self.transform_fn = transform
        self.target_transform_fn = target_transform
        self.dir_path = path
        self.data_file_name = data_file_name

        protocol, path = get_protocol_and_path(os.path.join(self.dir_path, self.data_file_name))
        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol)
        

    def _load(self) -> "KedroPytorchImageDataSet":
        self.data=pd.read_csv(os.path.join(self.dir_path, self.data_file_name))
        return self
        
        

    def _save(self, data: pd.DataFrame) -> None:
        self.data=data
        with self._fs.open(self._data_path, "w") as f:
            data.to_csv(f, index=False)
        
    @property
    def _data_path(self) -> str:
        return os.path.join(self.dir_path, self.data_file_name)

    def _describe(self) -> Dict[str, Any]:
        return {'directory': self.dir_path,
                'data_path': self._data_path,
                #'num_examples': self.data.shape[0],
                'transform': self.transform_fn,
                'target_transform': self.target_transform_fn}

    def __get_item__(self, index:int):
        assert index < self.data.shape[0], "Invalid index"
        label= self.data.loc[index,self.label_column]
        with self._fs.open(self.data.loc[index, self.path_column], "rb") as f:
            return self.transform_fn(Image.open(f).convert("RGB")), self.target_transform_fn(label)
    def __len__(self) -> int:
        return self.data.shape[0] if self.data is not None else 0
    def with_transforms(self, transform=None, target_transform=None) -> "KedroPytorchImageDataSet":
        self.transform_fn = transform
        self.target_transform_fn = target_transform
        return self
        
    def __add__(self, other: Any):
        pass