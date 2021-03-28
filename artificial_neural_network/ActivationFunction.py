from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def backward(self, input, cache):
        pass