from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

def cf_matrix(predictions, label_encoder):
    label_order = list(label_encoder.keys())

    cm = confusion_matrix(
        predictions['label'],
        predictions['prediction'],
        labels=label_order
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_order
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    ax.set_title("Confusion Matrix")

    return fig

def loss_history_plot(loss_history):
    loss_history=loss_history.groupby(['set','epoch']).agg(loss=('loss','mean')).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    for set_type in ['train', 'validation']:
        subset = loss_history[loss_history['set'] == set_type]
        ax.plot(subset['epoch'], subset['loss'], label=set_type)
    ax.set_title(f'Loss History')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    return fig
def get_sample_predictions(predictions, sample_size):
    predictions['classification_status']=np.where(predictions['label']==predictions['prediction'],'TP','FN')
    predictions=predictions.sample(frac=1)




    sample_predictions_1=predictions[predictions.groupby(['label','classification_status'])['label'].transform('rank', method='first')<=sample_size]

    sample_predictions_2=predictions[predictions['label']!=predictions['prediction']]


    sample_predictions_2=sample_predictions_2[sample_predictions_2.groupby(['prediction'])['prediction'].transform('rank', method='first')<=sample_size]
    sample_predictions_2['classification_status']='FP'

    sample_predictions_2=sample_predictions_2.rename(columns={'prediction':'folder'})
    sample_predictions_1=sample_predictions_1.rename(columns={'label':'folder'})

    sample_predictions=pd.concat([sample_predictions_1[['path','folder','classification_status']],
                                sample_predictions_2[['path','folder','classification_status']]]
                            )
    return sample_predictions

def save_sample_predictions(sample_predictions):
    for idx, row in sample_predictions.iterrows():
        image=Image.open(row['path']).convert("RGB")
        path=f"{row['folder']}/{row['classification_status']}/{row['path'].split('/')[-1]}"
        yield {path:image}
    
    return sample_predictions