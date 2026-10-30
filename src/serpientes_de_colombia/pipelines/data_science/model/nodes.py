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
from .utils import model_calibration, load_generators
from torchvision.models import get_model
def fxd_feature_extraction_from_backbone(train_image_dataset,validation_image_dataset, training_params, label_map, device, backbone_model):
    
    
    
    
    print('=='*20)
    print('=='*20)
    print('=='*20)
    print(f"Extracting features using {backbone_model}...")
    print('=='*20)
    print('=='*20)
    print('=='*20)

    training_generator,validation_generator=load_generators(train_image_dataset,validation_image_dataset, training_params, label_map, device)
    device = torch.device("cuda" if device=="cuda" and  torch.cuda.is_available() else "cpu")
    num_classes=len(label_map.keys())


    model_conv = get_model(backbone_model, weights='DEFAULT')
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Sequential(
                                    nn.Linear(num_ftrs, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(256, num_classes)
                                )
    for param in model_conv.parameters():
        param.requires_grad = False
    for param in model_conv.fc.parameters():
        param.requires_grad = True
    

    model_conv = model_conv.to(device)


    criterion = nn.CrossEntropyLoss()


    optimizer_conv = torch.optim.SGD( model_conv.fc.parameters(),
                                    lr=1e-3,
                                    momentum=0.9
                                    )

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


def finetune_from_backbone(train_image_dataset,validation_image_dataset, training_params, label_map, device, backbone_model):

    print('=='*20)
    print('=='*20)
    print('=='*20)
    print(f"Finetuning {backbone_model}...")
    print('=='*20)
    print('=='*20)
    print('=='*20)


    

    training_generator,validation_generator=load_generators(train_image_dataset,validation_image_dataset, training_params, label_map, device)
    device = torch.device("cuda" if device=="cuda" and  torch.cuda.is_available() else "cpu")
    num_classes=len(label_map.keys())


    model_conv = get_model(backbone_model, weights='DEFAULT')
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Sequential(
                                    nn.Linear(num_ftrs, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.3),
                                    nn.Linear(256, num_classes)
                                )
    for param in model_conv.parameters():
        param.requires_grad = True
    for param in model_conv.fc.parameters():
        param.requires_grad = True
    

    model_conv = model_conv.to(device)


    criterion = nn.CrossEntropyLoss()


    optimizer_conv = torch.optim.SGD( model_conv.fc.parameters(),
                                    lr=1e-3,
                                    momentum=0.9
                                    )

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