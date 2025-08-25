import torch
import time
def model_calibration(model, dataloader, criterion, optimizer, scheduler, device, phase='train',num_epochs=25):
    best_accuracy = 0.0
    since = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
    
        # Each epoch has a training and validation phase
        #for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode
    
        running_loss = 0.0
        running_corrects = 0
    
        # Iterate over data.
        for inputs, labels, path in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
    
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
    
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if phase == 'train':
            scheduler.step()
    
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_accuracy = running_corrects.double() / len(dataloader.dataset)
    
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_accuracy:.4f}')
    
            # deep copy the model
            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                
    
       
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    if phase == 'val':
        print(f'Best val Acc: {best_accuracy:4f}')
    
    # load best model weights
    return model