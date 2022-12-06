import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class MLPClassifier:
    EPS = 1e-7
    VAL_PERCENTAGE = 0.3
    EARLY_STOP_ITERATIONS = 100
    MIN_LR = 1e-5

    def __init__(self, x: np.ndarray, y: np.ndarray, layers_dims: list, learning_rate: float, batch_size: int = 16):

        self.batch_size = batch_size

        self.num_classes = len(np.unique(y))
        y = self._onehot(y)

        x = self._input_normalize(x)
        self.validation_set, self.train_setl, self.train_labels, self.validation_labels = None, None, None, None

        self._divide_input(x=x, y=y)

        self.layers_dims = layers_dims
        self.parameters = self.initialize_parameters()
        self.layers = len(self.parameters) // 2
        self.learning_rate = learning_rate
        self.validation_costs = []
        self.train_costs = []

    def initialize_parameters(self) -> (dict, dict):
        """
        Initialize the parameters (wheights and biases) according to layers_dims using kaiming initialization.
        :return: The parameters' dict.
        The weights of layer i is in key [W{i}]
        The bias of layer i is in key [b{i}]
        """
        params = {}

        for index, layer_dim in enumerate(self.layers_dims[1:]):
            prev_layer_dim = self.layers_dims[index]
            y = np.sqrt(2.0 / prev_layer_dim)
            params[f'W{index + 1}'] = np.random.randn(prev_layer_dim, layer_dim) * y
            params[f'b{index + 1}'] = np.zeros((1, layer_dim))

        return params

    def _input_normalize(self, x) -> np.ndarray:
        """
        normalize the input x
        :param x: The model input
        :return: The normalized input
        """
        x = x.reshape((x.shape[0], -1))
        mean = np.mean(x)
        std = np.std(x)
        x = (x - mean) / (std + self.EPS)

        return x

    def _onehot(self, y):
        """
        one hot encoder
        :param y: The ndarray to encode as one-hot
        """
        return np.eye(self.num_classes)[y]

    def _divide_input(self, x: np.ndarray, y) -> (np.ndarray, np.ndarray):
        """
        Divide the input data and labels to a validation set and a train set according to the validation percentage.
        The function updates the result in the class.
        :param x: The given input after normalize.
        :param y; The labels.
        :return: (train set, validation set, va)
        """
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.VAL_PERCENTAGE,
                                                          shuffle=True, random_state=8)

        self.train_set = np.split(x_train, len(x_train) // self.batch_size)
        self.validation_set = np.split(x_val, len(x_val) // self.batch_size)

        self.train_labels = np.split(y_train, len(y_train) // self.batch_size)
        self.validation_labels = np.split(y_val, len(y_val) // self.batch_size)

    def fit(self):
        """
        Performs the train loop.
        Each loop is over a single batch.
        Each iteration checks the result over the validation set and average the costs.
        In addition, the cost on the current training epoch is saved.
        When there is {EARLY_STOP_ITERATIONS} without any improvement -
        The learning rate is decreased by half until it reach {MIN_LR}
        """
        batch_index = 0
        iterations_without_improve = 0
        best_validation_cost = 100
        iteration = 0

        # The loop will stop when we reach {EARLY_STOP_ITERATIONS} without improvement with the
        # minimum learning rate allowed
        while iterations_without_improve < self.EARLY_STOP_ITERATIONS and self.learning_rate > self.MIN_LR:
            batch = self.train_set[batch_index]
            batch_labels = self.train_labels[batch_index]

            batch, caches = self.linear_model_forward(batch)
            grads = self.linear_model_backward(batch, batch_labels, caches)
            self.update_parameters(grads)

            cost = self.compute_cost(batch, batch_labels)
            self.train_costs.append(cost)

            print(f"Train costs on iteration {iteration} is {cost}")

            val_costs = []
            for val_batch, val_label in zip(self.validation_set, self.validation_labels):
                batch, _ = self.linear_model_forward(val_batch)
                cost = self.compute_cost(batch, val_label)
                val_costs.append(cost)

            cost = sum(val_costs) / len(val_costs)
            self.validation_costs.append(cost)

            print(f"Validation costs on iteration {iteration} is {cost}")

            if cost < best_validation_cost:
                best_validation_cost = cost
                iterations_without_improve = 0
            else:
                iterations_without_improve += 1
                print(f"{iterations_without_improve} iterations with no improve")

            if iterations_without_improve == self.EARLY_STOP_ITERATIONS:
                self.learning_rate /= 2
                iterations_without_improve = 0
                print(f"LEARNING RATE changed and is now {self.learning_rate}")

            batch_index = (batch_index + 1) % len(self.train_set)
            iteration += 1

    @staticmethod
    def linear_forward(x, weights, bias) -> (np.ndarray, dict):
        """
        Implement the linear part of a forward propagation
        :param x: The activations of the previous layer - or input data. (of shape [batch size, size of previous layer])
        :param weights: the weight matrix of the current layer
                        (of shape [size of previous layer, size of current layer])
        :param bias: the bias vector of the current layer (of shape [1, size of current layer])

        :return: (x, linear_cache)
        x – the linear component of the activation function (i.e., the value before applying the non-linear function)
            of shape [batch size, size of current layer]
        linear_cache – a dictionary containing x, W, b (stored for making the backpropagation easier to compute)
        """
        linear_cache = {'x': x,
                        'W': weights,
                        'b': bias}
        x = np.add(np.matmul(x, weights), bias)

        return x, linear_cache

    @staticmethod
    def softmax(x_batch: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Implement Softmax
        :param x_batch: the linear component of the activation function.
        (i.e., the value before applying the non-linear function)
        :return:(x, x_cache);
                x_batch – the activations of the layer
                x_cache – original x, which will be useful for the backpropagation
        """
        x_cache = x_batch.copy()
        z = x_batch - np.max(x_batch, axis=1).reshape((-1, 1))
        x_exp = np.exp(z)
        x_sums = np.sum(x_exp, axis=1).reshape((-1, 1))
        x_batch = x_exp / x_sums

        return x_batch, x_cache

    @staticmethod
    def relu(x_batch: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Implement Relu
        :param x_batch: the linear component of the activation function.
        (i.e., the value before applying the non-linear function)

        :return:(x, x_cache);
                x_batch – the activations of the layer
                x_cache – original x, which will be useful for the backpropagation
        """
        x_cache = x_batch.copy()
        x_batch = np.maximum(0, x_cache)

        return x_batch, x_cache

    def linear_activation_forward(self, x: np.ndarray, weights: np.ndarray, bias: np.ndarray, activation: str) \
            -> (np.ndarray, (dict, dict)):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer
        :param x: activations of the previous layer
        :param weights: the weights matrix of the current layer
        :param bias: the bias vector of the current layer
        :param activation: the activation function to be used (a string, either “softmax” or “relu”)

        :return:(x, cache)
        x – the activations of the current layer
        cache – a tuple of dictionaries containing both linear_cache and activation_cache
                (activation_cache, linear_cache)
        """

        x, linear_cache = self.linear_forward(x=x, weights=weights, bias=bias)
        activation_cache = {}

        if activation == 'softmax':
            x, activation_cache = self.softmax(x)
        elif activation == 'relu':
            x, activation_cache = self.relu(x)
        else:
            print(f"No such activation function. needs to be either 'softmax' or 'relu' and got {activation}")

        cache = ({'Z': activation_cache},
                 linear_cache)

        return x, cache

    def linear_model_forward(self, x) -> (np.ndarray, list):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation
        :param x: the data, numpy array of shape (input size, number of examples)

        :return: (x, caches)
        x – the last post-activation value
        caches – a list of all the cache objects generated by the linear_forward function
        """

        caches = []

        for layer in range(self.layers - 1):
            x_prev = x.copy()
            x, cache = self.linear_activation_forward(x_prev,
                                                      self.parameters[f'W{layer + 1}'],
                                                      self.parameters[f'b{layer + 1}'],
                                                      'relu')
            caches.append(cache)

        x, cache = self.linear_activation_forward(x,
                                                  self.parameters[f'W{self.layers}'],
                                                  self.parameters[f'b{self.layers}'],
                                                  'softmax')
        caches.append(cache)

        return x, caches

    @staticmethod
    def compute_cost(x, y) -> float:
        """
        Implement the cost function - categorical cross-entropy loss.
        :param x: probability vector corresponding to your label predictions, shape (number of examples, num_of_classes)
        :param y: the labels vector (i.e. the ground truth) (one-hot)
        :return: the labels vector (i.e. the ground truth)
        """
        examples, _ = x.shape

        return - np.sum(y * np.log(x)) / examples

    @staticmethod
    def linear_backward(dx, cache) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Implements the linear part of the backward propagation process for a single layer
        :param dx:  the gradient of the cost with respect to the linear output of the

         layer (layer l)
        :param cache: tuple of values (dx_prev, W, b) coming from the forward propagation in the current layer

        :return: (dx_prev, dW, db)
        dx_prev - Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as dx_prev
        dw - Gradient of the cost with respect to W (current layer l), same shape as W
        db - Gradient of the cost with respect to b (current layer l), same shape as b
        """
        weights, dx_prev, b = cache['W'], cache['x'], cache['b']

        dw = np.matmul(dx_prev.T, dx)
        dx_prev = np.matmul(dx, weights.T)
        db = np.sum(dx, axis=0, keepdims=True)

        return dx_prev, dw, db

    @staticmethod
    def relu_backward(dx, activation_cache: dict) -> np.ndarray:
        """
        Implements backward propagation for a ReLU unit
        :param dx: the post-activation gradient
        :param activation_cache: contains x (stored during the forward propagation)

        :return: dx – gradient of the cost with respect to x
        """
        x = activation_cache['Z']
        dx = dx.copy()
        dx[x <= 0] = 0

        return dx

    @staticmethod
    def loss_backward(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Loss backward
        :param x: the probabilities vector. shape ()
        :param y: The labels in one hot. shape (batch_size, num_classes)
        :return: The loss derivative. shape ()
        """
        dl = x - y

        return dl

    def softmax_backward(self, dl, activation_cache: dict) -> np.ndarray:
        """
        Implements backward propagation for a softmax unit
        :param dl: the post-activation gradient
        :param activation_cache:

        :return: dx – gradient of the cost with respect to x
        """
        z = activation_cache['Z']
        sm, _ = self.softmax(z)
        dsm = sm * (1. - sm)
        dl = dl * dsm

        return dl

    def linear_activation_backward(self, dx, cache: (dict, dict), activation: str) -> \
            (np.ndarray, np.ndarray, np.ndarray):
        """
        Implements the backward propagation for the LINEAR->ACTIVATION layer.
        The function first computes dZ and then applies the linear_backward function
        :param dx: post activation gradient of the current layer
        :param cache: contains both the linear cache and the activations cache
        :param activation: the activation function wanted

        :return: (dx_prev, dw, db)
        dx_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as dx
        dw – Gradient of the cost with respect to W (current layer l), same shape as w
        db – Gradient of the cost with respect to b (current layer l), same shape as b
        """
        activation_cache, linear_cache = cache
        dx_prev, dw, db = None, None, None

        if activation == 'relu':
            dx = self.relu_backward(dx, activation_cache)
            dx_prev, dw, db = self.linear_backward(dx, linear_cache)

        elif activation == 'softmax':
            dx = self.softmax_backward(dx, activation_cache)
            dx_prev, dw, db = self.linear_backward(dx, linear_cache)

        return dx_prev, dw, db

    def linear_model_backward(self, x, y, caches: list) -> dict:
        """
        Implement the backward propagation process for the entire network
        :param x: the probabilities vector, the output of the forward propagation (L_mo
        del_forward)
        :param y: the true labels vector (the "ground truth" - true classifications)
        :param caches:  list of caches containing for each layer: a. the linear cache; b. the activation cache
        :return:
        Grads - a dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        layers = len(caches)
        dx = self.loss_backward(x, y)
        current_cache = caches[layers - 1]
        grads[f'dA{layers}'], grads[f'dW{layers}'], grads[f'db{layers}'] = \
            self.linear_activation_backward(dx, current_cache, activation='softmax')

        for layer in reversed(range(layers - 1)):
            current_cache = caches[layer]
            grads[f'dA{layer + 1}'], grads[f'dW{layer + 1}'], grads[f'db{layer + 1}'] = \
                self.linear_activation_backward(grads[f'dA{layer + 2}'], current_cache, activation='relu')

        return grads

    def update_parameters(self, grads):
        """
        Updates parameters using gradient descent
        :param grads: a python dictionary containing the gradients (generated by linear_model_backward)
        """
        for layer in range(self.layers):
            self.parameters[f'W{layer + 1}'] = self.parameters[f'W{layer + 1}'] - \
                                               self.learning_rate * grads[f'dW{layer + 1}']

            self.parameters[f'b{layer + 1}'] = self.parameters[f'b{layer + 1}'] - \
                                               self.learning_rate * grads[f'db{layer + 1}']

    def predict(self, x, y) -> float:
        """
        The function receives an input data and the true labels and
        calculates the accuracy of the trained neural network on the data
        :param x: the input data, a numpy array of shape (height*width, number_of_examples)
        :param y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
        :return:
        accuracy – the accuracy measure of the neural net on the provided data
        (i.e. the percentage of the samples for which the correct label receives the highest confidence score).
        Use the softmax function to normalize the output values.
        """
        x = self._input_normalize(x)
        probs, _ = self.linear_model_forward(x)
        probs, _ = self.softmax(probs)
        predictions = np.argmax(probs, axis=1)
        accuracy = (predictions == y).sum() / len(y)

        return accuracy

    def loss_graph(self, validation: bool = False, both: bool = False):
        """
        This function creates a graph - costs on iterations
        :param validation: Whether to show the costs of the validation (or the train). Default: False
        :param both: If set to True, the function ignores the validation parameter and shows
                    the loss graph with both train and validation
        """
        if both:
            plt.plot(np.arange(len(self.train_costs)), self.train_costs, 'r')
            plt.plot(np.arange(len(self.train_costs)), self.validation_costs, 'b')
            plt.show()
        else:
            if not validation:
                plt.plot(np.arange(len(self.train_costs)), self.train_costs)
                plt.show()
            else:
                plt.plot(np.arange(len(self.validation_costs)), self.validation_costs)
                plt.show()
