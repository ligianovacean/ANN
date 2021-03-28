import numpy as np

from artificial_neural_network.ActivationFunction import ActivationFunction

class SoftMax(ActivationFunction):
    def forward(self, Z):
        # We subtract np.maz(Z) to avoid nan errors caused by FP limitations
        expZ = np.exp(Z - np.max(Z))
        output = expZ / expZ.sum(axis=0, keepdims=True)

        cache = Z

        return output, cache
    
    def backward(self, A, Y):
        dZ = A - Y

        return dZ