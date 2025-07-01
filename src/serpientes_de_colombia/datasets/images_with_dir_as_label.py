from pathlib import Path
from typing import Any, Dict
from kedro.io import AbstractDataset



class ImageWithDirAsLabelDataset(AbstractDataset):
   
    def __init__(self, filepath: str):
       self.filepath = filepath
        

    def _load(self) -> Dict:
        p=Path(self.filepath)
        return {"path":self.filepath, "label": p.parent.name}
        
        

    def _save(self, data: Any) -> None:
        raise DataSetError(
            "ImageWithDirAsLabelDataset is read-only and cannot be saved."
        )
        
        

    def _describe(self) -> Dict[str, Any]:
        return {"filepath": str(self.filepath)}
            