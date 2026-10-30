import pandas as pd
import torch
from torchvision import  transforms
from torch.utils.data import Dataset,  DataLoader

def get_predictions(data,model, label_map, training_params, device):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),   # same resizing, no randomness
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ])
    test_set = data.with_transforms(transform=transform, )

    test_generator = DataLoader(test_set, **training_params['generator_params'])

    model.eval()
    predictions = []
    labels = []
    paths = []
    inverse_label_map={v:k for k,v in label_map.items()}
    model=model.to(device)
    with torch.no_grad():
        for X, y, path in test_generator:
            X = X.to(device)
            preds = model(X).argmax(1).cpu()  # get predicted classes
            
            predictions.extend([ inverse_label_map[k.item()] for k in  preds])
            labels.extend(y)
            paths.extend(path)
    test_data_with_predictions=pd.DataFrame({'path':paths,
              'label': labels,
              'prediction': predictions
             })
    return test_data_with_predictions