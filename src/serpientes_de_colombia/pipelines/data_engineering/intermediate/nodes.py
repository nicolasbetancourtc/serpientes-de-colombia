from torchvision import datasets
import pandas as pd
import numpy as np

def train_validation_test_split(image_urls,   
train_size,     
validation_size):
    image_urls=image_urls[['observation_date','file_name','label']]


    image_urls=image_urls.sort_values(by='observation_date').reset_index(drop=True)
    train_validation, test= image_urls[image_urls.index<len(image_urls)*(1-validation_size)], image_urls[image_urls.index>=len(image_urls)*(1-validation_size)]

    train_validation['set']=train_validation.groupby('label')['label'].transform(lambda x: np.random.binomial(n=1,p=train_size, size=len(x)))


    train, validation= train_validation=train_validation[train_validation['set']==1], train_validation[train_validation['set']==0]
    return train, validation, test
