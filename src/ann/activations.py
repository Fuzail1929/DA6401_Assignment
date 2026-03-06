import numpy as np

# 1. Sigmoid Activation Function: The Sigmoid class implements the sigmoid activation function, which maps input values to a range between 0 and 1. 
class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

# gradient is sigmoid(x) * (1 - sigmoid(x))
    def backward(self, grad):
        return grad * self.out * (1 - self.out)


# 2. Tanh Activation Function: The Tanh class implements the hyperbolic tangent activation function, which maps input values to a range between -1 and 1. 
class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    
# gradient is 1 - tanh(x)^2
    def backward(self, grad):
        return grad * (1 - self.out * self.out)


# 3. ReLU Activation Function: The ReLU class implements the Rectified Linear Unit activation function, which outputs the max(0, Input Provided).
class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
# gradient is 1 for x > 0 and 0 for x <= 0
    def backward(self, grad):
        return grad * (self.x > 0)


