from pathlib import Path
from typing import Any, Dict
from kedro.io import AbstractDataset
from kedro_datasets.matplotlib import MatplotlibWriter



class LabeledImageDataset(MatplotlibWriter):
   
    def __init__(self, 
                 filepath: str, 
                 label: str,
                 **kwargs: Any
                 ):
       self.label = label

       
       super().__init__(filepath=filepath, **kwargs)
        

    def _load(self) -> Dict:
        super()._load()
        
        

    def _save(self, data: Any) -> None:
        if not Path(self.filepath.parent).exists():
            Path(self.filepath.parent).mkdir()
        super()._save(data)
        
        
        

    def _describe(self) -> Dict[str, Any]:
        return {"filepath": str(self.filepath),
                "label": self.label
                }
            