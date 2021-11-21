import collections
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math
import tensorflow as tf



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


class Datasets:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    def ALL_NUMBERS():
        return Datasets.x_train, Datasets.y_train, Datasets.x_test, Datasets.y_test

    def SIX_AND_NINES():
        x_train = Datasets.x_train[(Datasets.y_train == 6) | (Datasets.y_train == 9)]
        y_train = Datasets.y_train[(Datasets.y_train == 6) | (Datasets.y_train == 9)]
        x_test = Datasets.x_test[(Datasets.y_test == 6) | (Datasets.y_test == 9)]
        y_test = Datasets.y_test[(Datasets.y_test == 6) | (Datasets.y_test == 9)]
        return x_train, y_train, x_test, y_test

    def ONLY_SIXES():
        """
        only take 6
        expected to have the lowest complexity
        """
        x_train = Datasets.x_train[Datasets.y_train == 6]
        y_train = Datasets.y_train[Datasets.y_train == 6]
        x_test = Datasets.x_test[Datasets.y_test == 6]
        y_test = Datasets.y_test[Datasets.y_test == 6]
        return x_train, y_train, x_test, y_test

    def RANDOM():
        x_train = np.array([np.random.rand(Datasets.x_train[0].shape[0], Datasets.x_train[0].shape[1]) * 255 for _ in range(len(Datasets.x_train))]).astype(int)
        y_train = np.array([np.random.randint(0, 10) for _ in range(len(Datasets.x_train))])
        x_test = np.array([np.random.rand(Datasets.x_test[0].shape[0], Datasets.x_test[0].shape[1]) * 255 for _ in range(len(Datasets.x_test))]).astype(int)
        y_test = np.array([np.random.randint(0, 10) for _ in range(len(Datasets.x_test))])
        return x_train, y_train, x_test, y_test

    def SAMEPICTURE():
        x_train = np.array([Datasets.x_train[0] for _ in range(len(Datasets.x_train))])
        y_train = np.array([Datasets.y_train[0] for _ in range(len(Datasets.x_train))])
        x_test = np.array([Datasets.x_test[0] for _ in range(len(Datasets.x_test))])
        y_test = np.array([Datasets.y_test[0]for _ in range(len(Datasets.x_test))])
        return x_train, y_train, x_test, y_test

    def get_all_data(concat = False):
        all_num = Datasets.ALL_NUMBERS()
        six_nines = Datasets.SIX_AND_NINES()
        only_sixes = Datasets.ONLY_SIXES()
        random = Datasets.RANDOM()
        same_picture = Datasets.SAMEPICTURE()

        data = {
            "ALL_NUMBERS": {
                "x_train": all_num[0],
                "y_train": all_num[1],
                "x_test": all_num[2],
                "y_test": all_num[3]
            
            },
            "SIX_AND_NINES": {
                "x_train": six_nines[0],
                "y_train": six_nines[1],
                "x_test": six_nines[2],
                "y_test": six_nines[3]
            },
            "ONLY_SIXES": {
                "x_train": only_sixes[0],
                "y_train": only_sixes[1],
                "x_test": only_sixes[2],
                "y_test": only_sixes[3]
            },
            "RANDOM": {
                "x_train": random[0],
                "y_train": random[1],
                "x_test": random[2],
                "y_test": random[3]
            },
            "SAMEPICTURE": {
                "x_train": same_picture[0],
                "y_train": same_picture[1],
                "x_test": same_picture[2],
                "y_test": same_picture[3]
            }
        }

        if concat:
            new_data = {}
            for dataset in data:
                new_data[dataset] = dict()
                new_data[dataset]["x"] = np.concatenate((data[dataset]["x_train"],data[dataset]["x_test"]))
                new_data[dataset]["y"] = np.concatenate((data[dataset]["y_train"],data[dataset]["y_test"]))
            return new_data
        else: return data    
       
  
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

    def PCA(data,reshape=True,n_components=2):
        """
        Performs PCA on data.
        """
        data_dim = data.shape[1]*data.shape[2]
        pca = PCA(n_components=n_components)
        data_transformed = pca.fit_transform(data.reshape(data.shape[0],data_dim))
        if reshape:
            return data_transformed.reshape(data.shape[0],int(math.sqrt(n_components)),int(math.sqrt(n_components)))
        else: return data_transformed


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

