from torchvision import datasets
import pandas as pd

def list_train_files(train_image_list):
    results=[]
    for partition_key, partition_load_function in train_image_list.items():
        file_path = partition_load_function()
        results.append(file_path)
    print(results[10])
    return pd.DataFrame(results)
