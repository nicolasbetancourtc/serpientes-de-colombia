import numpy as np
def train_validation_test_split(image_urls,   
train_validation_size,     
validation_size):
    min_train_fraction=0.8*image_urls['label'].value_counts().min()

    train_validation=image_urls[image_urls.assign(random=np.random.normal(0,1)).groupby('label')['random'].transform('rank','first')<=min_train_fraction]
    test=image_urls[~image_urls['observation_id'].isin(train_validation['observation_id'])]

    train_validation['set']=train_validation.groupby('label')['label'].transform(lambda x: np.random.binomial(n=1,p=validation_size, size=len(x)))

    train, validation=train_validation[train_validation['set']==0], train_validation[train_validation['set']==1]
    print(f"Train set size: {len(train)}")
    print(f"Validation set size: {len(validation)}")    
    print(f"Test set size: {len(test)}")
    print(f"Train set class distribution:\n{train['label'].value_counts(normalize=True)}")
    return train, validation, test



