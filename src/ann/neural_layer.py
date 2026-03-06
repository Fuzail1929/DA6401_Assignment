import numpy as np

class NeuralLayer:

#  Constructor of the class which initializes the weights and biases of the layer. 
#  We tried to Weights are initialized using either Xavier initialization or a small random normal distribution, while biases are initialized to zero.
   
    def __init__(self, in_dim, out_dim, init="xavier"):

        if init == "xavier":
            limit = np.sqrt(6 / (in_dim + out_dim))
            self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        elif init == "zeros":
            self.W = np.zeros((in_dim, out_dim))
        else:
            self.W = np.random.randn(in_dim, out_dim) * 0.01

        self.b = np.zeros((1, out_dim))

# 1. Initialisation of grad_weight and grad_bias at a neural layer.
        self.grad_W = None
        self.grad_b = None

# 2. Forward pass through the layer, We then computed the output as a linear transformation of the input.
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

# 3. Backward Propagation : We get the gradients with respect to the weights and biases & also calculated the gradient with respect to the input .
    def backward(self, grad_out):
        self.grad_W = self.x.T @ grad_out
        self.grad_b = np.sum(grad_out, axis=0, keepdims=True)

        grad_input = grad_out @ self.W.T
        return grad_input