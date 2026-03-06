import numpy as np

'''
    1. Mean Squared Error (MSE): 

    -> The MSE class implements the mean squared error loss function,commonly used for regression tasks.
    
    -> It calculates the average of the squared differences between the predicted values and the true values, 

'''

class MSE:
    def forward(self, y_pred, y_true):

        if y_true.ndim == 1:
            one_hot = np.zeros_like(y_pred)
            one_hot[np.arange(len(y_true)), y_true] = 1
            y_true = one_hot

        self.y_pred = y_pred
        self.y_true = y_true

        return np.mean((y_pred - y_true) ** 2)

# gredient is 2 * (y_pred - y_true) / n where n is the number of samples in the batch 
    def backward(self):
        return 2 * (self.y_pred - self.y_true) / len(self.y_true)


'''
    2. Cross-Entropy Loss: 
        
        -> The CrossEntropy class implements the cross-entropy loss function, commonly used for classification tasks. 
         
        -> It calculates the loss based on the predicted probabilities and the true class labels 
        and also computes the gradient for backpropagation.
'''

class CrossEntropy:
    def forward(self, logits, y_true):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.y_true = y_true

        logp = -np.log(self.probs[np.arange(len(y_true)), y_true])
        return np.mean(logp)

# gadient is (p−y)/N (predicted probabilities minus true labels).
    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(len(self.y_true)), self.y_true] -= 1
        return grad / len(self.y_true)