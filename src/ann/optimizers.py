import numpy as np

''' 1. SGD: 
       It will updates model parameters by taking small steps in the direction that reduces the loss, 
       taking only one datapoint at a time, using the gradient of the error.
'''

class SGD:
    def step(self, layer, lr, wd):
        layer.W -= lr * (layer.grad_W + wd * layer.W)
        layer.b -= lr * layer.grad_b



'''
    2. Momentum : 
        adds a velocity (running average of gradients) so updates are smoother instead of jumping randomly like sgd.

'''

class Momentum:
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.vW = {}
        self.vb = {}

    # Updating layer’s weights and biases using the momentum update rule 

    def step(self, layer, lr, wd):
        k = id(layer)

        if k not in self.vW:
            self.vW[k] = np.zeros_like(layer.W)
            self.vb[k] = np.zeros_like(layer.b)

        # calculating gradient with weight decay if wd > 0, 
        # otherwise using the original gradient for weight updates.

        if wd > 0:
            grad_W = layer.grad_W + wd * layer.W
        else:
            grad_W = layer.grad_W

        grad_b = layer.grad_b

        self.vW[k] = self.gamma * self.vW[k] + lr * grad_W
        self.vb[k] = self.gamma * self.vb[k] + lr * grad_b

        layer.W -= self.vW[k]
        layer.b -= self.vb[k]



'''
      3. Nesterov Accelerated Gradient (NAG): 
            Updates parameters using momentum with a look-ahead step, 
            so it try to anticipates the future position and makes faster, more accurate updates.
'''

class NAG:
    # gamma = momentum coefficient
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.vW = {}
        self.vb = {}
    
    # Updates parameters by combining previous and current velocity for gradient calculation 
    def step(self, layer, lr, wd):
        k = id(layer)

        # initialize velocity
        if k not in self.vW:
            self.vW[k] = np.zeros_like(layer.W)
            self.vb[k] = np.zeros_like(layer.b)

        # first , perform look - ahead step 
        W_look = layer.W - self.gamma * self.vW[k]
        b_look = layer.b - self.gamma * self.vb[k]

        # gradients should be computed at look-ahead point
        if wd > 0:
            grad_W = layer.grad_W + wd * W_look
        else:
            grad_W = layer.grad_W

        grad_b = layer.grad_b

        # updating velocity for both parameters using the computed gradients
        self.vW[k] = self.gamma * self.vW[k] + lr * grad_W
        self.vb[k] = self.gamma * self.vb[k] + lr * grad_b

        # updating the parameters  
        layer.W -= self.vW[k]
        layer.b -= self.vb[k]

'''
        4. RMSProp: 
            A method that updates the layer’s weights and biases using gradients to reduce training loss efficiently
'''
class RMSProp:

    # gamma: moving average coefficient
    # eps : numerical stability

    def __init__(self, gamma=0.9, eps=1e-8):
        self.gamma = gamma
        self.eps = eps
        self.sW = {}
        self.sb = {}

    # Updates parameters using adaptive learning rates based on squared gradients for stable training
    
    def step(self, layer, lr, wd):
        k = id(layer)

        # initialize cache
        if k not in self.sW:
            self.sW[k] = np.zeros_like(layer.W)
            self.sb[k] = np.zeros_like(layer.b)

        # gradients with weight decay if wd > 0, otherwise using original gradients for updates

        if wd > 0:
            grad_W = layer.grad_W + wd * layer.W
        else:
            grad_W = layer.grad_W

        grad_b = layer.grad_b

        # RMSProp update for both weights and biases, 
        # which involves updating the running average of squared gradients and then adjusting the parameters accordingly

        self.sW[k] = self.gamma * self.sW[k] + (1 - self.gamma) * (grad_W ** 2)
        self.sb[k] = self.gamma * self.sb[k] + (1 - self.gamma) * (grad_b ** 2)

        # parameter update
        layer.W -= lr * (grad_W / np.sqrt(self.sW[k] + self.eps))
        layer.b -= lr * (grad_b / np.sqrt(self.sb[k] + self.eps))



