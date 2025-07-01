from torchvision import datasets
import pandas as pd

def list_train_files(train_image_list):
    results=[]
    for partition_key, partition_load_function in train_image_list.items():
        file_path = partition_load_function()
        results.append(file_path)
    print(results[10])
    return pd.DataFrame(results)

def temp(train_data):
    print(train_data.data.shape)
    print(train_data.dir_path)
    print(train_data._data_path)
    print(train_data._describe())
    train_data.transform_fn
    train_data.target_transform_fn
    return train_data
