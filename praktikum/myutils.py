import collections
from genericpath import exists
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as np
import os
import json
import itertools
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model


class QuatumCircuit:

    def __init__(self):

        def variational_training_circuit(params, data):
            qml.templates.embeddings.AngleEmbedding(
                features=data, wires=range(wires), rotation="X"
            )
            return variational_circuit(params)

        def variational_circuit(params):
            for layer in range(layers):
                for wire in range(wires):
                    qml.RX(params[layer][wire][0], wires=wire)
                    qml.RY(params[layer][wire][1], wires=wire)
                for wire in range(0, wires - 1, 2):
                    qml.CZ(wires=[wire, wire + 1])
                for wire in range(1, wires - 1, 2):
                    qml.CZ(wires=[wire, wire + 1])
            return qml.expval(qml.PauliZ(0))

        # fixed variables
        wires = 4
        layers = 4

        # init params
        self.parameters = np.random.uniform(low=-np.pi, high=np.pi, size=(layers, wires, 2))
        dev = qml.device('default.qubit', wires=wires, shots=1000)

        self.training_circuit = qml.QNode(func=variational_training_circuit, device=dev)

    # some helpers
    def correctly_classified(self, params, data, target):
        prediction = self.training_circuit(params, data)
        if prediction < 0 and target[0] > 0:
            return True
        elif prediction > 0 and target[1] > 0:
            return True
        return False

    def overall_cost_and_correct(self, cost_fn, params, data, targets):
        cost = correct_count = 0
        for datum, target in zip(data, targets):
            cost += cost_fn(params, datum, target)
            correct_count += int(self.correctly_classified(params, datum, target))
        return cost, correct_count

    def distributed_cost(self, params, data, target):
        """Cost function distributes probabilities to both classes."""
        prediction = self.training_circuit(params, data)
        scaled_prediction = prediction + 1 / 2
        predictions = np.array([1 - scaled_prediction, scaled_prediction])
        return np.sum(np.abs(target - predictions))

    def cost(self, params, data, target):
        """Cost function penalizes choosing wrong class."""
        prediction = self.training_circuit(params, data)
        predictions = np.array([0, prediction]) if prediction > 0 else np.array([prediction * -1, 0])
        return np.sum(np.abs(target - predictions))

    def train(self,x_train,y_train,x_test,y_test,epochs):

        optimizer = qml.AdamOptimizer()
        cost_fn = self.distributed_cost

        hist = dict()

        hist['accuracy'] = []
        hist['val_accuracy'] = []
        hist['loss'] = []
        hist['val_loss'] = []
        for epoch in range(epochs):

            training_correct_count = 0
            training_cost = 0
            for datum, target in zip(x_train, y_train):
                self.parameters = optimizer.step(lambda weights: cost_fn(weights, datum, target), self.parameters)

                training_cost += cost_fn(self.parameters, datum, target)
                training_correct_count += int(self.correctly_classified(self.parameters, datum, target))


            test_cost, test_correct_count = self.overall_cost_and_correct(cost_fn, self.parameters, x_test, y_test)

            hist['accuracy'].append(training_correct_count / len(x_train))
            hist['val_accuracy'].append(test_correct_count / len(x_test))
            hist['loss'].append(training_cost / len(x_train))
            hist['val_loss'].append(test_cost / len(x_test))

            print("epoch {}: train_cost:{:.3f} train_acc:{:.3f} test_cost:{:.3f}  test_acc:{:.3f}".
            format(epoch, hist["loss"][-1],hist["accuracy"][-1],hist["val_loss"][-1],hist["val_accuracy"][-1]))

        return hist

    def predict(self,x_test):
        prediction = self.training_circuit(self.parameters, x_test)

        if prediction < 0:
            return [1, 0]
        else:
            return [0, 1]
        





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

    def Autoencoder(x_train,x_test,outputsize=4,epochs=6):
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
                    layers.Dense(16, activation='sigmoid'),
                    layers.Dense(392, activation='sigmoid'),
                    layers.Dense(784, activation='sigmoid') # output layer
                ])
                self.compile(optimizer='adam', loss=losses.MeanSquaredError())
                
            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        tf.random.set_seed(42)

        autoencoder = AutoencoderTF(latent_dim=outputsize)

        x_train = x_train / 255
        x_test = x_test / 255

        tf.compat.v1.reset_default_graph()
        with tf.device('/gpu:0'):
            hist = autoencoder.fit(x_train, x_train, epochs=epochs,verbose=0,batch_size=32).history

        x_train_auto = autoencoder.encoder(x_train).numpy()
        x_test_auto = autoencoder.encoder(x_test).numpy()

        del autoencoder

        return x_train_auto, x_test_auto , hist

class Complexity_Measures:
    def neuralnetwork(x_train, y_train, x_test, y_test, epochs=3, batch_size=32):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        with tf.device('/gpu:0'):
            tf.compat.v1.reset_default_graph()
            tf.keras.backend.clear_session()

            model = tf.keras.Sequential([
                tf.keras.layers.Dense(4, activation='relu'),
                tf.keras.layers.Dense(2, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam',
                                loss='binary_crossentropy',
                                metrics=['accuracy'])
            hist = model.fit(x_train, y_train, epochs=epochs, batch_size=32,validation_data=(x_test, y_test),verbose=0)   

            del model

            return {
                "train_accuracy":hist.history["val_accuracy"],
                "train_loss":hist.history["val_loss"],
                "test_accuracy":hist.history["val_accuracy"],
                "test_loss":hist.history["val_loss"]

            }

    def get_measures(*args):
        result = dict()

        for preprocessing in os.listdir("data/"):
        
            for subset in os.listdir("data/"+preprocessing+"/"):

                if os.path.exists("data/"+preprocessing+"/"+subset+"/measures.json"):
                    with open("data/"+preprocessing+"/"+subset+"/measures.json") as json_file:
                        measures = json.load(json_file)
                    
                    for measure in measures:
                        if measure not in result:
                            result[measure] = dict()
                            result[measure][preprocessing] = dict()
                            result[measure][preprocessing][subset] = measures[measure]
                        else:
                            if preprocessing not in result[measure]:
                                result[measure][preprocessing] = dict()
                            result[measure][preprocessing][subset] = measures[measure]

        return result

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

        x = Helpers.normalize(x,min=0,max=255).astype("int")
    
        
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
        
        return 1/ (1+np.amax(fdr))

class Datasets:

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    def get_preprocessed_datasets(*args):

        #load the datasets
        data = dict()

        requested_datasets = args
        if args == ():
            requested_datasets = os.listdir('data')

        #build dict holding all data
        for preprocessing in requested_datasets:

            if preprocessing not in os.listdir('data'):
                print(preprocessing, "not found")
                continue

            if preprocessing not in data:
                data[preprocessing] = dict()

            for dataset in os.listdir('data/' + preprocessing):
                if dataset not in data[preprocessing]:
                    data[preprocessing][dataset] = dict()

                for type in os.listdir('data/' + preprocessing + '/' + dataset):
                    if type.split(".")[1] == "npy":
                        data[preprocessing][dataset][type.split(".")[0]] = np.load('data/' + preprocessing + '/' + dataset + '/' + type)

        return data

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

    def reshape(data, shape):
        reshape_shape = (data.shape[0],) + shape
        print("reshape_shape", reshape_shape)
        return data.reshape(reshape_shape)

    def write_measure2(measure,value,preprocessing,subset,config=None,runiteration=-1):
        if not os.path.isdir("measure/" + preprocessing):
            os.mkdir("measure/" + preprocessing)

        if not os.path.exists("measure/" + preprocessing + "/" + subset+".json"):
            with open("measure/" + preprocessing + "/" + subset+".json", "w") as f:
                json.dump({}, f)

        
        with open("measure/" + preprocessing + "/" + subset+".json", "r+") as f:
            data = json.load(f)

            f.truncate()



            if config is None:
                data[measure] = value
                json.dump(data, f)
                return


            if measure not in data:
                data[measure] = {}
            data[measure][str(config)] = value
            json.dump(data, f)
    
    def write_measure(measure,value,preprocessing,subset,runiteration=-1):
        if not os.path.isdir("measure/" + preprocessing):
            os.mkdir("measure/" + preprocessing)

        if not os.path.exists("measure/"+preprocessing+"/"+subset+".json"):
            with open("measure/"+preprocessing+"/"+subset+".json","w") as f:
                if runiteration > -1:
                    value = [value]
                json.dump({
                    measure: value
                },f,indent=4)
        else:
            d = json.loads(open("measure/"+preprocessing+"/"+subset+".json").read())

            if runiteration > -1:
                if runiteration == 0:
                    d[measure] = []
                d[measure].append(value)
            else: d[measure] = [value]

            with open("measure/"+preprocessing+"/"+subset+".json","w") as f:
                json.dump(d,f,indent=4)

    def normalize(data,min=0,max=255):
        """
        Normalizes data between min and max.
        """
        data_norm = data-(np.min(data))
        data_norm = data_norm / ( np.max(data_norm) / (max - min))
        data_norm = data_norm+min
        return data_norm

    def plot_grid(data, labels=None ,rows=2, cols=5):
        """
        Plots a grid of data.
        """
        root = int(math.sqrt(data.shape[1]))
        data = data.reshape(data.shape[0], root, root)

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

    def store(data,folder,filename):

        if not os.path.isdir(folder):
            os.makedirs(folder)

        if type(data) == np.ndarray:
            np.save(folder+"/"+filename+".npy",data)

        elif type(data) == dict:
            with open(folder+"/"+filename+".json","w") as f:
                json.dump(data,f)

        else:
            print("Type not supported")