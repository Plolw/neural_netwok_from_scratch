import numpy as np

def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Sigmoid_derivative(z):
    s = Sigmoid(z)
    return s * (1 - s)

def ReLU(z):
    return np.maximum(0, z)

def ReLU_derivative(z):
    return (z > 0).astype(float)

def Softmax(z):
    max_z = np.max(z, axis=1, keepdims=True)
    ez = np.exp(z - max_z)
    
    sum_ez = np.sum(ez, axis=1, keepdims=True)
    return ez / sum_ez