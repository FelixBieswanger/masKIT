import collections
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math


class Complexity_Measures:

    def entropy(data):
        """
        Calculates the (shannon) entropy of a data set.
        """

        #reshape the data to a 2D array
        dim = data.shape[1]*data.shape[2]
        data_reshape = data.reshape(data.shape[0],dim)

        #calculate the entropy
        entropy = np.zeros(dim)

        #iterating over each pixel
        for i in range(dim):
            #calculate the probability of each pixel value
            _, counts = np.unique(data_reshape[:, i], return_counts=True)
            entropy[i] = np.sum(-counts / len(data_reshape[:, i])
                                 * np.log(counts / len(data_reshape[:, i])))

        #returning the entropy of each pixel as np.array                                
        return entropy
        

    def method2(data):
        return 3


class Helpers:

    def normalize(data,min=0,max=255):
        """
        Normalizes data between min and max.
        """
        data_min = np.min(data)
        data_max = np.max(data)
        data_normalized = (data-data_min)/(data_max-data_min)*(max-min)+min
        return data_normalized.astype(int)

    def plot_grid(data, labels=None ,rows=2, cols=5):
        """
        Plots a grid of data.
        """
        fig, ax = plt.subplots(rows, cols, figsize=(15, 8))
        for i in range(rows):
            for j in range(cols):
                ax[i, j].imshow(data[i * cols + j], cmap='gray')
                ax[i, j].set_title(labels[i * cols + j])
                ax[i, j].axis('off')
        plt.show()

    def PCA(data, n_components=2):
        """
        Performs PCA on data.
        """
        data_dim = data.shape[1]*data.shape[2]
        pca = PCA(n_components=n_components)
        data_transformed = pca.fit_transform(data.reshape(data.shape[0],data_dim))
        return data_transformed.reshape(data.shape[0],int(math.sqrt(n_components)),int(math.sqrt(n_components)))


    def plot_classify_results(predictions,labels):
        """
        Plots the confusion matrix & accuracy.
        """
        cm = confusion_matrix(labels, predictions, labels=[6, 9])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[6, 9])
        disp.plot()

        corrects = 0
        for i in range(len(predictions)):
            if predictions[i] == labels[i]:
                corrects += 1
        print("Accurracy",corrects / len(predictions))
