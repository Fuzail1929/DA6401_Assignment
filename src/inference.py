
import argparse
import numpy as np
import wandb

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from ann.neural_network import NeuralNetwork
from ann.objective_functions import CrossEntropy, MSE
from utils.data_loader import load_data
from collections import Counter


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument("-m", "--model_path", type=str, default="best_model.npy")

    parser.add_argument("-d", "--dataset",
                        choices=["mnist", "fashion_mnist"],
                        default="mnist",
                        )

    parser.add_argument("-b", "--batch_size",
                        default=64,
                        type=int)

    parser.add_argument("-sz", "--hidden_layers",
                        nargs="+", 
                        type=int, 
                        default = [128]
                        )

    parser.add_argument("-nhl", "--num_neurons",
                        type=int,
                        default  = 1
                        )

    parser.add_argument("-a", "--activation",
                    nargs="+",
                    choices=["relu", "sigmoid", "tanh"],
                    default = 'relu'
                    )
    
    parser.add_argument("-o", "--optimizer",
                        choices=["sgd", "momentum", "nag", "rmsprop"],
                        default = "rmsprop",
                        )

    parser.add_argument("-l", "--loss",
                        choices=["cross_entropy", "mse"],
                        default = "cross_entropy",
                        )

    parser.add_argument("-w_i", "--weight_init",
                        choices=["random", "xavier"],
                        default = "random",
                        )

    parser.add_argument("-lr", "--learning_rate",
                        type=float,
                        default=0.01,
                        )

    parser.add_argument("-wd", "--weight_decay",
                        type=float,
                        default=0)
    
    parser.add_argument("--wandb_project",
                    type=str,
                    default="dl_assignment")
    
    parser.add_argument("--wandb_group",
                    type=str,
                    default="experiment_1")

    return parser.parse_args()


def load_model(model_path):
    """
        -> Load trained model from disk.
    """

    weights = np.load(model_path, allow_pickle=True)
    return weights



def evaluate_model(model, X_test, y_test):
    """
       -> Evaluate model on test data.
    """
    
    logits = model.forward(X_test)

    preds = np.argmax(logits, axis=1)
           

    
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    precision = precision_score(y_test, preds, average="macro")
    recall = recall_score(y_test, preds, average="macro")

    loss_fn = CrossEntropy()
    loss = loss_fn.forward(logits, y_test)

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }



def main():
    args = parse_arguments()

    args.hidden_size = args.hidden_layers
    args.num_layers = len(args.hidden_layers)

    wandb.init(
    project=args.wandb_project,
    group=args.wandb_group,
    name="inference"  
)

    if len(args.activation) == 1:
        args.activation = args.activation * args.num_layers


    _, _, X_test, y_test = load_data(args.dataset)

    class DummyArgs:
        pass

    cli_args = DummyArgs()
    cli_args.hidden_size = args.hidden_layers
    cli_args.num_layers = len(args.hidden_layers)
    cli_args.activation = args.activation
    cli_args.weight_init = args.weight_init
    cli_args.loss = args.loss
    cli_args.optimizer = args.optimizer
    cli_args.learning_rate = args.learning_rate
    cli_args.weight_decay = args.weight_decay

    model = NeuralNetwork(cli_args)

    weights = load_model(args.model_path)

    for i, layer in enumerate(model.layers):
        layer.W, layer.b = weights[i]

    results = evaluate_model(model, X_test, y_test)


    print("Loss:", results["loss"])
    print("Accuracy:", results["accuracy"])
    print("F1:", results["f1"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])

    print("Evaluation complete!")
    
    wandb.finish()
    return results
    


if __name__ == '__main__':
    main()
