import numpy as np
def train_validation_test_split(image_urls,   
train_validation_size,     
validation_size):
   # Number of samples on each class
   min_train_fraction=train_validation_size*image_urls['label'].value_counts().min()
   # 1. Assign a random number to each row
   # 2. Order the rows based on the random number within each label group
   # 3. Use the first train_validation_size % (e.g. 80%) as train_validation set
   train_validation=image_urls[image_urls.assign(random=np.random.uniform(0,1)).groupby(['label'])['random'].transform('rank','first')<=min_train_fraction]
   # Test are the set of all observations outside train or validation (to prevent leackage at observation level)
   test=image_urls[~image_urls['observation_id'].isin(train_validation['observation_id'])]

   # Number of samples on each class
   min_validation_fraction=validation_size*train_validation['label'].value_counts().min()
   # 1. Assign a random number to each row
   # 2. Order the rows based on the random number within each label group
   # 3. Use the first train_validation_size % (e.g. 80%) as train_validation set
   validation=train_validation[train_validation.assign(random=np.random.uniform(0,1)).groupby(['label'])['random'].transform('rank','first')<=min_validation_fraction]
   # Train are the set of all observations outside validation (to prevent leackage at observation level)
   train=train_validation[~train_validation['observation_id'].isin(validation['observation_id'])]
   print(f"Train set size: {len(train)}")
   print(f"Validation set size: {len(validation)}")    
   print(f"Test set size: {len(test)}")
   print(f"Train set class distribution:\n{train['label'].value_counts(normalize=True)}")
    
   
 
   return train, validation, test



