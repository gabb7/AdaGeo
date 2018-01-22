"""
Classes that implement the Optimizable abstract class present in
base_classes.py. These are used to show how the AdaGeo optimization framework
can be used.

Gabriele Abbati, Machine Learning Research Group, University of Oxford
January 2018
"""


# Libraries
from adageo.base_classes import Optimizable
import numpy as np
from random import sample
# Dataset
from keras.datasets import mnist
# Keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as k_back
from keras.callbacks import Callback


class LossHistory(Callback):
    """
    Loss function history class, used to monitor the performances of the
    optimizer. Inherits from the Keras class Callback.
    """

    def __init__(self):
        """
        Constructor.
        """
        Callback.__init__(self)
        self.losses = None
        return

    def on_train_begin(self, logs=None):
        """
        Initializes the list of loss function values.
        """
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        """
        Updates the list of loss function values.
        """
        self.losses.append(logs.get('loss'))


class LogRegressionMNIST(Optimizable):
    """
    Implementation of a one-layer neural network to perform logistic
    regression on the MNIST dataset.
    """

    def __init__(self, n_batch: int = 128):
        """
        Constructor.
        :param n_batch: number of elements in the batch used to compute the
        gradients with back-propagation.
        """
        self.n_batch = n_batch
        self.n_classes = 10
        self.img_rows, self.img_cols = 28, 28
        self.input_size = self.img_rows * self.img_cols
        self.x_train, self.y_train, self.n_train, \
            self.x_test, self.y_test, self.n_test =\
            None, None, None, None, None, None
        self.model = None
        self.weights = None
        self.weights_shapes = None
        self.gradients = None
        self.get_keras_gradients = None
        self.history = None
        self.test_history = []
        self.training_history = []
        self.build_dataset()
        return

    def build_dataset(self) -> None:
        """
        Downloads the MNIST dataset if needed and format it (normalization and
        formatting of the y vectors).
        """
        # Load the MNIST dataset and reshape it accordingly to the model
        (self.x_train, self.y_train), (self.x_test, self.y_test) =\
            mnist.load_data()
        self.n_train = self.x_train.shape[0]
        self.n_test = self.x_test.shape[0]
        self.x_train = self.x_train.reshape(self.x_train.shape[0],
                                            self.input_size)
        self.x_test = self.x_test.reshape(self.x_test.shape[0],
                                          self.input_size)
        # Normalize
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0
        # Convert class vectors to binary class matrices
        self.y_train = to_categorical(self.y_train, self.n_classes)
        self.y_test = to_categorical(self.y_test, self.n_classes)
        return

    def build_network(self) -> None:
        """
        Builds the network architecture using Keras utilities and compiles it.
        One hidden layer implementing logistic regression.
        """
        self.model = Sequential()
        self.model.add(Dense(self.n_classes, activation='sigmoid',
                             input_dim=self.x_train.shape[1]))
        self.model.add(Dropout(0.7))
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd',
                           metrics=['accuracy'])
        return

    def build_weight_buffers(self) -> None:
        """
        Builds some buffers used to access the weights of the network and the
        shapes of the layers.
        """
        self.weights = self.model.trainable_weights
        self.weights_shapes = []
        for w_vec in self.weights:
            self.weights_shapes.append(w_vec.shape)
        return

    def build_gradient_buffers(self) -> None:
        """
        Builds some buffers that will be used to compute the gradients of the
        loss function with respect to the weights in the network.
        """
        self.gradients =\
            self.model.optimizer.get_gradients(self.model.total_loss,
                                               self.weights)
        input_tensors = [self.model.inputs[0],                      # input data
                         self.model.sample_weights[0],  # weight for each sample
                         self.model.targets[0],                         # labels
                         k_back.learning_phase()]           # train or test mode
        self.get_keras_gradients = k_back.function(inputs=input_tensors,
                                                   outputs=self.gradients)
        return

    def build_model(self) -> None:
        """
        Builds the whole model by invoking the previous auxiliary functions.
        """
        self.build_network()
        self.build_weight_buffers()
        self.build_gradient_buffers()
        self.history = LossHistory()
        self.test_history = []
        self.training_history = []
        return

    def next_train_batch(self) -> [np.array, np.array]:
        """
        Extracts a new batch from the training set.
        :return: list with two elements: the x values of the batch and the
        corresponding targets.
        """
        index = sample(range(self.n_train), self.n_batch)
        x_sample = self.x_train[index, :]
        y_sample = self.y_train[index, :]
        return [x_sample, y_sample]

    def next_test_batch(self) -> [np.array, np.array]:
        """
        Extracts a new batch from the test set.
        :return: list with two elements: the x values of the batch and the
        corresponding targets.
        """
        index = sample(range(self.n_test), self.n_batch)
        x_sample = self.x_test[index, :]
        y_sample = self.y_test[index, :]
        return [x_sample, y_sample]

    def train(self, n_epochs: int = 5) -> None:
        """
        Trains the neural network for a fixed number of epochs using the
        internal optimizers of Keras.
        :param n_epochs: number of training epochs
        """
        self.model.fit(self.x_train, self.y_train, epochs=n_epochs,
                       batch_size=self.n_batch, callbacks=[self.history])
        return

    def predict(self, x_new: np.array) -> np.array:
        """
        Predicts the label for the input x_new using the current state of the
        network.
        :param x_new: numpy array with dimensions
        [n_items, self.img_rows, self.img_cols] containing the test points.
        :return: numpy array containing the predicted labels.
        """
        return self.model.predict(x_new, batch_size=self.n_batch)

    def compute_test_loss(self) -> None:
        """
        Computes the value of the loss function on a randomly chosen test batch.
        """
        [x_batch, y_batch] = self.next_test_batch()
        loss = self.model.test_on_batch(x_batch, y_batch)
        self.history.losses.append(loss)
        self.test_history.append(loss)
        return

    def compute_training_loss(self) -> None:
        """
        Computes the value of the loss function on a randomly chosen test batch.
        """
        [x_batch, y_batch] = self.next_train_batch()
        loss = self.model.test_on_batch(x_batch, y_batch)
        self.history.losses.append(loss)
        self.training_history.append(loss)
        return

    def set_weight_list(self, weights: list) -> None:
        """
        Sets the weights in the network with the ones passed as argument.
        :param weights: list of numpy arrays, one for each layer, containing
        the values of the weights.
        """
        self.model.set_weights(weights)
        return

    def get_weight_vector(self) -> np.array:
        """
        Gets the values of the weights of the network, re-formatted as a 1-D
        vector.
        :return: 1-D numpy array containing the value of the weights.
        """
        weights = self.model.get_weights()
        weights_vector = weights[0].flatten()
        n_layers = len(weights)
        for n in range(1, n_layers):
            weights_vector = np.append(weights_vector, weights[n].flatten())
        return weights_vector

    def set_weight_vector(self, weights: np.array) -> None:
        """
        Takes the weights vector passed as argument, converts it to a list of
        elements with the right shapes and sets the new weights of the network.
        :param weights: 1-D vector containing the weight values.
        """
        processed_size = 0
        new_weights_list = []
        for tensor_shape in self.weights_shapes:
            present_size = int(np.prod(tensor_shape))
            layer_weights = np.copy(weights[processed_size:processed_size +
                                            present_size])
            new_weights_list.append(np.reshape(layer_weights, tensor_shape))
            processed_size = processed_size + present_size
        self.model.set_weights(new_weights_list)
        return

    def get_gradient_vector(self) -> np.array:
        """
        Compute the gradient of the loss function of the curent weights and
        returns it as a 1D numpy array.
        :return: 1D numpy array containing the numerical value of the gradient.
        """
        [x_batch, y_batch] = self.next_train_batch()
        inputs = [x_batch, np.ones(self.n_batch), y_batch, 0]
        grads = self.get_keras_gradients(inputs)
        grad_vector = grads[0].flatten()
        n_layers = len(grads)
        for n in range(1, n_layers):
            grad_vector = np.append(grad_vector, grads[n].flatten())
        return grad_vector

    def f(self, x: np.array) -> float:
        """
        Returns the value of the training loss function of the network computed
        with the weights in x.
        :param x: numpy array containing the weights of the network.
        :return: value of the objective function at x.
        """
        self.set_weight_vector(x)
        self.compute_training_loss()
        return self.training_history[-1]

    def get_gradient(self, x: np.array) -> np.array:
        """
        Returns the gradients of the objective function we want to optimize
        computed at x.
        :param x: numpy array containing the desired coordinates.
        :return: function gradients at x.
        """
        self.set_weight_vector(x)
        gradient = self.get_gradient_vector()
        return gradient
