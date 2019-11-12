import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(x, y, cm, classes, cmap=plt.cm.Greens):
    plt.figure(figsize=(x, y))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title('Confusion matrix', size=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes)+1)-0.5
    
    plt.xticks(tick_marks, classes, horizontalalignment="left")
    plt.yticks(tick_marks, classes, rotation=90)

    thresh = (cm.max()+cm.min()) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center", 
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylim(9.5, -0.5)
    plt.grid()
    plt.ylabel('True label', size=20)
    plt.xlabel('Predicted label', size=20)
    plt.tight_layout()
