import numpy as np


class Standardization:
    def __init__(self, input):
        n = len(input)
        self.mu = (1 / n) * np.sum(input)
        self.dev = np.sqrt((1 / n) * np.sum((input - self.mu) ** 2))

    def Scaler(self, input):
        return (input - self.mu) / self.dev

    def ReScaler(self, input):
        return self.mu + self.dev * input


class minmaxNormalization:
    def __init__(self, input):
        self.min = np.min(input)
        self.max = np.max(input)

    def Scaler(self, input):
        return (input - self.min) / (self.max - self.min)

    def ReScaler(self, input):
        return self.min + input * (self.max - self.min)
