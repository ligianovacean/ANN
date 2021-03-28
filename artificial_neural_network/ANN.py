import numpy as np

from artificial_neural_network.ANNLayer import ANNLayer
from artificial_neural_network.ReLU import ReLU
from artificial_neural_network.SoftMax import SoftMax


class ANN:
    def __init__(self, layers_dims):
        self.layers = self.initialize_parameters(layers_dims)
        self.caches = []
        self.grads = {}


    def initialize_parameters(self, layer_dims):
        network_layers = []

        parameters = {}
        L = len(layer_dims)          

        # First L-1 layers have ReLU activation
        for i in range(1, L-1):
            network_layers.append(ANNLayer(layer_dims[i-1], layer_dims[i], ReLU()))

        # Last layer (output layer) has Sigmoid activation
        network_layers.append(ANNLayer(layer_dims[L-2], layer_dims[L-1], SoftMax()))

        return network_layers


    def forward(self, X):
        self.caches = []
        A = X
        L = len(self.layers)

        for i in range(L):
            A_prev = A
            A, cache = self.layers[i].forward(A_prev)
            self.caches.append(cache)

        return A


    def backward(self, result, Y):
        self.grads = {}
        L = len(self.layers)
        m = result.shape[1]
        Y = Y.reshape(result.shape) 

        # Softmax activation in last layer
        (linear_cache, activation_cache) = self.caches[L-1]
        current_cache = (linear_cache, Y)
        self.grads["dA" + str(L-1)], self.grads["dW" + str(L)], self.grads["db" + str(L)] = self.layers[L-1].backward(result, current_cache)

        for i in reversed(range(L-1)):
            current_cache = self.caches[i]
            dA_prev_temp, dW_temp, db_temp = self.layers[i].backward(self.grads['dA'+str(i+1)], current_cache)

            self.grads["dA" + str(i)] = dA_prev_temp
            self.grads["dW" + str(i + 1)] = dW_temp
            self.grads["db" + str(i + 1)] = db_temp


    def compute_loss(self, logits, Y):
        m = Y.shape[1]

        # Cross-entropy loss in log scale
        loss = - np.sum(Y * np.log(logits + 1e-8)) / m

        return loss


    def update_parameters(self, learning_rate):
        L = len(self.layers)

        for i in range(L):
            self.layers[i].weight = self.layers[i].weight - learning_rate*self.grads['dW'+str(i+1)]
            self.layers[i].bias = self.layers[i].bias - learning_rate*self.grads['db'+str(i+1)]


    def fit(self, X, Y, x_test, y_test, iters, learning_rate, patience=50, do_early_stopping=True, verbose=True):
        train_losses = []
        test_losses = []

        i = 0
        do_stop = False
        while i < iters and not do_stop:
            # Forward pass
            result = self.forward(X)

            # Compute train loss
            loss = self.compute_loss(result, Y)
            train_losses.append(loss)

            # Backward pass
            self.backward(result, Y)

            # Parameters update
            self.update_parameters(learning_rate)

            # Compute test loss
            test_result = self.forward(x_test)
            test_loss = self.compute_loss(test_result, y_test)
            test_losses.append(test_loss)

            if verbose:
                print(f"Iter {i}: Train loss = {loss}, Validation loss = {test_loss}")

            if do_early_stopping and i > patience:
                threshold = test_losses[i - patience]
                latest_test_losses = np.array(test_losses[i - patience + 1: i + 1])

                if np.all(latest_test_losses >= threshold):
                    do_stop = True

            i += 1

        return train_losses, test_losses


    def predict(self, X, Y):
        m = X.shape[1]
        n = len(self.layers)
        p = np.zeros((1,m))

        prediction = self.forward(X)

        #Softmax
        y_hat = np.argmax(prediction, axis=0)
        Y = np.argmax(Y, axis=0)

        return y_hat, Y

        