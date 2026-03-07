"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import os 
import numpy as np
import wandb
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')

    # Dataset
    parser.add_argument("-d", "--dataset",
                        type=str,
                        required=True,
                        default = "mnist" ,
                        choices=["mnist", "fashion_mnist"])

    # Training hyperparameters
    parser.add_argument("-e", "--epochs",
                        type=int,
                        default = '5',
                        required=True)

    parser.add_argument("-b", "--batch_size",
                        type=int,
                        default = 64,
                        required=True)

    parser.add_argument("-lr", "--learning_rate",
                        type=float,
                        default = '0.01',
                        required=True)

    parser.add_argument("-wd", "--weight_decay",
                        type=float,
                        default =0.0,
                        )

    # Optimizer
    parser.add_argument("-o", "--optimizer",
                        type=str,
                        required=True,
                        default = "rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])

    # Architecture
    parser.add_argument("-nhl", "--num_layers",
                        type=int,
                        default ="1",
                        required=True)

    parser.add_argument("-sz", "--hidden_size",
                        type=int,
                        nargs="+",
                        required=True)

    parser.add_argument("-a", "--activation",
                        type=str,
                        nargs="+",
                        required=True,
                        default='relu',
                        choices=["relu", "sigmoid", "tanh"])

    # Loss
    parser.add_argument("-l", "--loss",
                        type=str,
                        required=True,
                        default = "cross_entropy",
                        choices=["cross_entropy", "mse"])

    # Weight init
    parser.add_argument("-w_i", "--weight_init",
                        type=str,
                        required=True,
                        default = "xavier",
                        choices=["random", "xavier"])

    # W&B
    parser.add_argument("--wandb_project",
                    type=str,
                    default="dl_assignment"
                    )

    # Model save path (relative)
    parser.add_argument("--model_save_path",
                        type=str,
                        default="best_model.npy")

    parser.add_argument(
                "--wandb_group",
                type=str ,
                default="experiment_1"
)
    return parser.parse_args()



def main():
    args = parse_arguments()
    
    if len(args.activation) == 1:
        args.activation = args.activation * args.num_layers

    if len(args.hidden_size) != args.num_layers:
        raise ValueError("hidden_size list must match num_layers")
    
    if len(args.activation) != args.num_layers:
        raise ValueError("activation list must match num_layers")

    wandb.init(
    project=args.wandb_project,
    group=args.wandb_group,
)   


    X_train, y_train, _, _ = load_data(args.dataset)
    
    class DummyArgs:
        pass

    cli_args = DummyArgs()
    cli_args.dataset = args.dataset
    cli_args.epochs = args.epochs
    cli_args.batch_size = args.batch_size
    cli_args.loss = args.loss
    cli_args.optimizer = args.optimizer
    cli_args.learning_rate = args.learning_rate
    cli_args.weight_decay = args.weight_decay
    cli_args.num_layers = args.num_layers
    cli_args.hidden_size = args.hidden_size
    cli_args.activation = args.activation
    cli_args.weight_init = args.weight_init

    model = NeuralNetwork(cli_args)
    

    model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    

    weights = model.get_weights()

    dir_path = os.path.dirname(args.model_save_path)

    if dir_path != "":
        os.makedirs(dir_path, exist_ok=True)

    np.save(args.model_save_path, weights, allow_pickle=True)
    
    wandb.finish()

    print("Training complete!")

if __name__ == '__main__':
    main()
