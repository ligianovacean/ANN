import numpy as np

class ANNLayer:
    def __init__(self, n_in, n_out, activation):
        self.weight = np.random.randn(n_out, n_in) * 0.1
        self.bias = np.zeros((n_out, 1))
        self.activation = activation


    def linear_forward(self, input_data):
        output = np.dot(self.weight, input_data) + self.bias

        cache = (input_data, self.weight, self.bias)

        return output, cache

    
    def forward(self, input_data):
        Z, linear_cache = self.linear_forward(input_data)
        A, activation_cache = self.activation.forward(Z)
        cache = (linear_cache, activation_cache)

        return A, cache


    def linear_backward(self, dZ, cache):
        (A_prev, W, b) = cache
        m = A_prev.shape[1]

        dW = (1/m) * np.dot(dZ, A_prev.T) 
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db


    def backward(self, dA, cache):
        (linear_cache, activation_cache) = cache
        
        dZ = self.activation.backward(dA, activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db