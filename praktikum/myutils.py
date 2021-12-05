import collections
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math
import tensorflow as tf
import os
import json
import itertools
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model



class Preprocessing:

    def minmax_scaler(data,min=0,max=1):
        scaler = MinMaxScaler(feature_range=(min, max))
        return scaler.fit_transform(data)

    
    def preprocessing_methods():
        return zip([Preprocessing.PCA, Preprocessing.Autoencoder,Preprocessing.Raw] , ["PCA", "Autoencoder","RAW"])

    def Raw(x_train,x_test,outputsize=4):
        return x_train, x_test


    def PCA(x_train,x_test,outputsize=4):
        pca = PCA(n_components=outputsize)
        pca.fit(x_train)
        x_train_pca = pca.transform(x_train)
        x_test_pca = pca.transform(x_test)

        return x_train_pca, x_test_pca

    def Autoencoder(x_train,x_test,outputsize=4,epochs=3):

        #mute tensorflow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        class AutoencoderTF(Model):
            def __init__(self, latent_dim):
                super(AutoencoderTF, self).__init__()
                self.encoder = tf.keras.Sequential([
                    layers.Dense(784, activation='relu'),
                    layers.Dense(392, activation='relu'),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(latent_dim, activation='relu'),
                ])
                self.decoder = tf.keras.Sequential([
                    layers.Dense(latent_dim,activation='sigmoid'),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(392, activation='relu'),
                    layers.Dense(784, activation='relu') # output layer
                ])
                self.compile(optimizer='adam', loss=losses.MeanSquaredError())
                

            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        autoencoder = AutoencoderTF(latent_dim=outputsize)

        x_train = x_train / 255
        x_test = x_test / 255

        autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=32)

        x_train_auto = autoencoder.encoder(x_train).numpy()
        x_test_auto = autoencoder.encoder(x_test).numpy()
        return x_train_auto, x_test_auto


class NeuralNetwork:
    
    def run(preprocessing, x_train, y_train, x_test, y_test, run_number=3, epochs=3, batch_size=32):

        #mute tensorflow logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        #set-up result storing
        if os.path.exists("results/neuralnetwork/"+preprocessing+".json"):
            os.remove("results/neuralnetwork/"+preprocessing+".json")

        for run in range(run_number):

            print(run, "of", run_number)

            for subset in itertools.combinations([i for i in range(10)],2):
                
                x_train_subset = x_train[(y_train == subset[0]) | (y_train == subset[1])]
                y_train_subset = y_train[(y_train == subset[0]) | (y_train == subset[1])]
                y_train_subset = np.where(y_train_subset == subset[0], 0, y_train_subset)
                y_train_subset = np.where(y_train_subset == subset[1], 1, y_train_subset)

                x_test_subset = x_test[(y_test == subset[0]) | (y_test == subset[1])]
                y_test_subset = y_test[(y_test == subset[0]) | (y_test == subset[1])]
                y_test_subset = np.where(y_test_subset == subset[0], 0, y_test_subset)
                y_test_subset = np.where(y_test_subset == subset[1], 1, y_test_subset)

                if preprocessing == "pca":
                    pca = PCA(n_components=4)
                    x_train_subset = pca.fit_transform(x_train_subset)
                    x_test_subset = pca.transform(x_test_subset)

                    x_train_subset = Helpers.normalize(x_train_subset,min=0,max=1,dtype=np.float32)
                    x_test_subset = Helpers.normalize(x_test_subset,min=0,max=1,dtype=np.float32)

                elif preprocessing == "raw":
                    x_train_subset = Helpers.normalize(x_train_subset,min=0,max=1,dtype=np.float32)
                    x_test_subset = Helpers.normalize(x_test_subset,min=0,max=1,dtype=np.float32)

                with tf.device('/gpu:0'):
                    tf.keras.backend.clear_session()

                    model = tf.keras.Sequential([
                        tf.keras.layers.Dense(4, activation='relu'),
                        tf.keras.layers.Dense(2, activation='relu'),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])

                    model.compile(optimizer='adam',
                                        loss='binary_crossentropy',
                                        metrics=['accuracy'])
                    hist = model.fit(x_train_subset, y_train_subset, epochs=3, batch_size=32,validation_data=(x_test_subset, y_test_subset),verbose=0)

                    Helpers.log_results(filename="results/neuralnetwork/"+preprocessing+".json", result={
                        str(run): {
                            str(subset):hist.history["val_accuracy"][-1]
                        } 
                    })

                    del model

    



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
            entropy[i] = -np.sum(counts / len(data_reshape[:, i])
                                 * np.log2(counts / len(data_reshape[:, i])))

        #returning the entropy of each pixel as np.array                                
        return entropy
        

    def fischer_discriminat_ratio(x, y):
        #calculate the fisher discriminant ratio of data  
        unique_y = np.unique(y)
    
        
        classes = {}
        for label in unique_y:
            classes[label] = x[y == label]
        
        fdr = np.zeros(x.shape[1])
        for fi in range(4):
        
            zähler = list()
            for j in unique_y:
                for k in unique_y:
                    if j != k:
                        p_cj = len(classes[j]) / len(x)
                        p_ck = len(classes[k]) / len(x)
        
                        u_cj = np.mean(classes[j][:,fi], axis=0)
                        u_ck = np.mean(classes[k][:,fi], axis=0)
        
                        zähler.append(p_cj*p_ck*(np.power((u_cj-u_ck),2)))
        
        
            nenner = list()
            for j in unique_y:
                p_cj = len(classes[j]) / len(x)
                sigma_cj = np.std(classes[j][:,fi], axis=0)
                nenner.append(p_cj*(np.power(sigma_cj,2)))
        
            if np.sum(nenner) == 0.0:
                fdr[fi] = 0
            else:
                fdr[fi] = np.sum(zähler)/np.sum(nenner)
        
        return fdr

class Datasets:

    def get_preprocessed_datasets():
        data = dict()

        #build dict holding all data
        for preprocessing in os.listdir('data'):

            if preprocessing not in data:
                data[preprocessing] = dict()

            for dataset in os.listdir('data/' + preprocessing):
                if dataset not in data[preprocessing]:
                    data[preprocessing][dataset] = dict()

                for type in os.listdir('data/' + preprocessing + '/' + dataset):
                    data[preprocessing][dataset][type.split(".")[0]] = np.load('data/' + preprocessing + '/' + dataset + '/' + type)

        return data


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
        x_test = np.array([Datasets.x_train[0] for _ in range(len(Datasets.x_test))])
        y_test = np.array([Datasets.y_train[0] for _ in range(len(Datasets.x_test))])
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
                print(dataset)
                new_data[dataset]["x"] = np.concatenate((data[dataset]["x_train"],data[dataset]["x_test"]))
                new_data[dataset]["y"] = np.concatenate((data[dataset]["y_train"],data[dataset]["y_test"]))
            return new_data
        else: return data    
       
  
class Helpers:

    def log_results(result, filename):
        if not os.path.exists(filename):
            with(open(filename,"w")) as f:
                f.write(json.dumps(result))
        else:
            with(open(filename,"r")) as f:
                data = json.load(f)
        
            for key in result:
                if key in data:
                    data[key].update(result[key])
                else:
                    data.update(result)

            with(open(filename,"w")) as f:
                f.write(json.dumps(data))


    def normalize(data,min=0,max=255,):
        """
        Normalizes data between min and max.
        """
        data_norm = data-(np.min(data))
        print(np.max(data_norm) / (max - min))
        data_norm = data_norm / ( np.max(data_norm) / (max - min))
        data_norm = data_norm+min
        return data_norm

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

