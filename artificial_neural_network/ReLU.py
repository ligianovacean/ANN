import numpy as np

from artificial_neural_network.ActivationFunction import ActivationFunction

class ReLU(ActivationFunction):
    def forward(self, Z):
        A = np.maximum(0, Z)
        cache = Z 
        
        return A, cache

    def backward(self, dA, cache):
        dZ = dA.copy()
        Z = cache
        dZ[Z < 0] = 0

        return dZ