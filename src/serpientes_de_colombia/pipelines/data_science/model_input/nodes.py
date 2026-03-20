import numpy as np
import pandas as pd

def global_label_encoder(image_dataset):
    image_data= image_dataset.data
    label_map=dict(
        zip(
            image_data[image_dataset.label_column].unique(),
            range(image_data[image_dataset.label_column].nunique())
        )
        )
    return label_map
