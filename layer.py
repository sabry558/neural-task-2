import numpy as np

class Layer:
    def __init__(self, neurons, activation_function, learning_rate, weight_num):
        self.neurons = int(neurons)
        self.activation_function = activation_function
        self.learning_rate = float(learning_rate)
        self.weights_num = weight_num
        self.differentiating = np.zeros(self.neurons)

        self.W = np.random.uniform(low=-1.0, high=1.0, size=(self.neurons, self.weights_num))

        self.bias = np.zeros(self.neurons)

        self.a_out = np.zeros(self.neurons)

        self.error = np.zeros(self.neurons)

        
    def hyperbolic_tangent(self, Z):
        return np.tanh( Z)
    
    def hyperbolic_tangent_differentiating(self, Z):
        return (1 - np.tanh(Z) ** 2)

    def activation(self, Z, index):
        if self.activation_function.lower() == "sigmoid":
            self.differentiating[index] = self.sigmoid_differentiating(Z)
            return self.sigmoid(Z)
        elif self.activation_function.lower() == "hyperbolic_tangent":
            self.differentiating[index] = self.hyperbolic_tangent_differentiating(Z)
            return self.hyperbolic_tangent(Z)
       

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(- Z))

    def sigmoid_differentiating(self, Z):
        y = self.sigmoid(Z)
        return y * (1 - y)

