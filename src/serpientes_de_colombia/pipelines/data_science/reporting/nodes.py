from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def cf_matrix(predictions, label_encoder):
    # Compute confusion matrix
    #print('a',type(predictions))
    #print('a',predictions)
    print('holaaaaa')
    cm = confusion_matrix(predictions['label'], predictions['prediction'])
    # Plot it
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.keys())
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    ax.set_title("Confusion Matrix")
    return fig