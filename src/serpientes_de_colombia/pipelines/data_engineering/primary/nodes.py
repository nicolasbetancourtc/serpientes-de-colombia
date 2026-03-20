import numpy as np
def train_validation_test_split(image_urls,   
train_validation_size,     
validation_size):
    image_urls=image_urls[['observation_date','file_name','label']]

    #Sort by observation date to avoid data leakage. Test will be the most recent data.
    image_urls=image_urls.sort_values(by='observation_date', ascending=True).reset_index(drop=True)
    #Test will be the most recent data.
    train_validation, test= image_urls[image_urls.index<len(image_urls)*train_validation_size], image_urls[image_urls.index>=len(image_urls)*train_validation_size]
    #Within train_validation, split into train and validation
    train_validation['set']=train_validation.groupby('label')['label'].transform(lambda x: np.random.binomial(n=1,p=validation_size, size=len(x)))


    train, validation=train_validation[train_validation['set']==0], train_validation[train_validation['set']==1]
    return train, validation, test



