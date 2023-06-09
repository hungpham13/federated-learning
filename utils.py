from matplotlib import pyplot as plt
import numpy as np
import itertools
from config import CLASSES

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    return figure

def plot_tensorboard(tensorboard_writer, loss, accuracy, precision, confusion_matrix, prefix, step):
    tensorboard_writer.add_scalar(f"Loss/{prefix}", loss, step)
    tensorboard_writer.add_scalar(f"Accuracy/{prefix}", accuracy, step)
    tensorboard_writer.add_figure(f"Confusion Matrix/{prefix}", plot_confusion_matrix(confusion_matrix, CLASSES), step)
    for label in precision:
        if precision[label] is not None:
            tensorboard_writer.add_scalar(f"Precision {label}/{prefix}", precision[label], step)
