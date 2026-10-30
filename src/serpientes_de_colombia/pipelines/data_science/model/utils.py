import torch
import time
import pandas as pd
from torchvision import  transforms
from torch.utils.data import DataLoader




def load_generators(train_image_dataset,validation_image_dataset, training_params, label_map, device):
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
        shuffle=True,
        num_workers=training_params['generator_params'].get("num_workers", 0)
    )
    validation_generator = DataLoader(
        validation_set, 
        shuffle=False,
    **training_params['generator_params']
    )
    device = torch.device("cuda" if device=="cuda" and  torch.cuda.is_available() else "cpu")
    return training_generator, validation_generator


def model_calibration(model, train_dataloader, validation_dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    loss_history = []
    for epoch in range(num_epochs):

    
        model.eval()
        model.fc.train()
        #model.train()

        train_loss = 0.0
        for idx, (x, y, _) in enumerate(train_dataloader):   # <- ignore path with _
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)
            loss_history.append({'loss':loss.item(), 'batch': idx, 'epoch': epoch, 'set': 'train'})


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #train_loss += loss.item()

        #train_loss /= len(train_dataloader)

        # ---- Validation ----
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            correct=0
            total=0
            for idx, (x, y, _) in enumerate(validation_dataloader):  
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)
                loss_history.append({'loss':loss.item(), 'batch': idx, 'epoch': epoch, 'set': 'validation'})
                
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                #val_loss += loss.item()

        #val_loss /= len(validation_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}")
    return model, pd.DataFrame(loss_history)
                
    
       
   