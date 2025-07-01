import numpy as np
import pandas as pd

def train_test_split(image_dataset,train_size):
    data=image_dataset.data
    data['set']=data.groupby(image_dataset.label_column).transform(lambda x: np.random.binomial(n=1,p=train_size, size=len(x)))
    data_train=data[data['set']==1]
    data_test=data[data['set']==0]
    return data_train.drop(columns=['set']), data_test.drop(columns=['set'])

