import torch
import time
import pandas as pd
def model_calibration(model, train_dataloader, validation_dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    loss_history = []
    for epoch in range(num_epochs):

        # ---- Train (backbone eval, head train) ----
        model.eval()
        model.fc.train()

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
            for idx, (x, y, _) in enumerate(validation_dataloader):  # <- ignore path here too
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)
                loss_history.append({'loss':loss.item(), 'batch': idx, 'epoch': epoch, 'set': 'validation'})

                #val_loss += loss.item()

        #val_loss /= len(validation_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}")
    return model, pd.DataFrame(loss_history)
                
    
       
   