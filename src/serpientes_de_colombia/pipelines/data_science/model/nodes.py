import numpy as np
import pandas as pd
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import  transforms
from PIL import Image
from .utils import model_calibration
from torchvision.models import get_model
from torch.utils.data import WeightedRandomSampler
def fxd_feature_extractor(train_image_dataset,validation_image_dataset, training_params, label_map, device, backbone_model):
    print('=='*20)
    print('=='*20)
    print('=='*20)
    print(f"Extracting features using {backbone_model}...")
    print('=='*20)
    print('=='*20)
    print('=='*20)


    train_transform = transforms.Compose([
    transforms.Resize((224, 224)),   # resize WITHOUT cropping
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),   # small rotations
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
      ])

    val_transform = transforms.Compose([
    transforms.Resize((224, 224)),   # same resizing, no randomness
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ])
    target_transform=lambda x: label_map.get(x)
    
    training_set = train_image_dataset.with_transforms(
        transform=train_transform, 
        target_transform=target_transform
        )
    validation_set = validation_image_dataset.with_transforms(
        transform=val_transform, 
        target_transform=target_transform
        )
   

    training_generator = DataLoader(
        training_set,
        batch_size=training_params['generator_params']["batch_size"],
        #sampler=sampler,
        num_workers=training_params['generator_params'].get("num_workers", 0)
    )
    validation_generator = DataLoader(validation_set, **training_params['generator_params'])
    device = torch.device("cuda" if device=="cuda" and  torch.cuda.is_available() else "cpu")

    model_conv = get_model(backbone_model)
    num_ftrs = model_conv.fc.in_features
    num_classes=train_image_dataset.data[train_image_dataset.label_column].nunique()
    model_conv.fc = nn.Sequential(
                                    nn.Linear(num_ftrs, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(256, num_classes)
                                )
    for param in model_conv.parameters():
        param.requires_grad = True
    

    model_conv = model_conv.to(device)


    criterion = nn.CrossEntropyLoss()


    optimizer_conv = torch.optim.SGD(model_conv.parameters(), lr=0.01, momentum=0.9)

    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    # model, train_dataloader, validation_dataloader, criterion, optimizer
    model_conv,loss_history=model_calibration(
        model_conv, 
        training_generator, 
        validation_generator, 
        criterion, 
        optimizer_conv,
        device, 
        num_epochs=training_params["num_epochs"]
        )
    
    return model_conv, loss_history