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
def fxd_feature_extractor(train_image_dataset,test_image_dataset, training_params, label_map, device, backbone_model):
    print('=='*20)
    print('=='*20)
    print('=='*20)
    print(f"Extracting features using {backbone_model}...")
    print('=='*20)
    print('=='*20)
    print('=='*20)


    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    target_transform=lambda x: label_map.get(x)
    
    training_set = train_image_dataset.with_transforms(transform=transform, target_transform=target_transform)
    test_set = test_image_dataset.with_transforms(transform=transform, target_transform=target_transform)
    
    training_generator = DataLoader(training_set, **training_params)
    test_generator = DataLoader(test_set, **training_params)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    model_conv = get_model(backbone_model)
    num_ftrs = model_conv.fc.in_features
    num_classes=train_image_dataset.data[train_image_dataset.label_column].nunique()
    model_conv.fc = nn.Linear(num_ftrs, num_classes)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()


    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


    model_conv=model_calibration(model_conv, training_generator, criterion, optimizer_conv, exp_lr_scheduler, device, phase='train',num_epochs=25)
    model_conv=model_calibration(model_conv, test_generator, criterion, optimizer_conv, exp_lr_scheduler, device, phase='val',num_epochs=25)
    return model_conv