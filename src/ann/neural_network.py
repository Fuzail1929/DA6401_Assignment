import numpy as np

import wandb
from .neural_layer import NeuralLayer
from .activations import Sigmoid, Tanh, ReLU
from .objective_functions import CrossEntropy, MSE
from .optimizers import SGD, Momentum, NAG, RMSProp


class NeuralNetwork:
    
   # Main model class that orchestrates the neural network training and inference.
    
    def __init__(self, cli_args):

        
        #first Initialize the neural network.
        #Args:  cli_args: Command-line arguments for configuring the network 
    
        self.args = cli_args
        act_map = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "tanh": Tanh,
        }
        opt_map = {
            "sgd": SGD,
            "momentum": Momentum,
            "nag": NAG,
            "rmsprop": RMSProp,
        }
        hidden = cli_args.hidden_size

        if not isinstance(hidden, list):
            hidden = [hidden]

        sizes = [784] + hidden + [10]
        #sizes = [784] + cli_args.hidden_size + [10]

        self.layers = []
        self.activations = []

        # Now , creates all network layers with hidden activations 
        # and selects the loss functions based on the CLI command-line arguments

        for i in range(len(sizes) - 1):
            self.layers.append(
                NeuralLayer(sizes[i], sizes[i+1], cli_args.weight_init)
            )
            if i < len(sizes) - 2:
                act = cli_args.activation
                if isinstance(act, str):
                    act = [act] * (len(sizes) - 2)
                self.activations.append(act_map[act[i]]())

        self.loss_fn = (
            CrossEntropy()
            if cli_args.loss == "cross_entropy"
            else MSE()
        )

        self.optimizer = opt_map[cli_args.optimizer]()
        self.lr = cli_args.learning_rate
        self.weight_decay = cli_args.weight_decay




    def forward(self, X):
        
        # Forward propagation through all layers.
        out = X

        for i in range(len(self.layers) - 1):
            out = self.layers[i].forward(out)
            out = self.activations[i].forward(out)

        logits = self.layers[-1].forward(out)
        return logits



    def backward(self, y_true, y_pred):
        
        # Backward propagation to compute gradients.
        
        self.loss_fn.forward(y_pred, y_true)
        grad = self.loss_fn.backward()

        for i in reversed(range(len(self.layers))):
            if i < len(self.activations):
                grad = self.activations[i].backward(grad)
            grad = self.layers[i].backward(grad)

        grad_w = [l.grad_W for l in self.layers]
        grad_b = [l.grad_b for l in self.layers]

        return grad_w, grad_b



    def update_weights(self):
        
        # Update weights using the optimizer.
        
        for layer in self.layers:
            self.optimizer.step(
                layer,
                self.args.learning_rate,
                self.args.weight_decay
            )



    def train(self, X_train, y_train, epochs, batch_size):
        
        # Train the network for specified epochs.
        
        n = len(X_train)

        for epoch in range(epochs):

            idx = np.random.permutation(n)
            X_train = X_train[idx]
            y_train = y_train[idx]

            epoch_loss = 0
            batches = 0

            

            for i in range(0, n, batch_size):

                xb = X_train[i:i+batch_size]
                yb = y_train[i:i+batch_size]

                y_pred = self.forward(xb)

                loss = self.loss_fn.forward(y_pred, yb)

                self.backward(yb, y_pred)

                self.update_weights()

                epoch_loss += loss
                batches += 1

            
            train_acc = self.evaluate(X_train, y_train)

            print(f"Epoch {epoch+1} | " f"Train Acc: {train_acc:.4f} | ")
            


    # Evaluation of model :  

    def evaluate(self, X, y):

        # Evaluate the network on given data.
    
        logits = self.forward(X)
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y)
        return acc
    

    # Utility methogs to safely export and restore model parameters layer-wise.

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d



    def set_weights(self, weight_dict):

        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
