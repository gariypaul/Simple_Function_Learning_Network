##############################################################
# Simple Function Reproduction using a shallow neural network#
# Command Line Application                                   #
# Author: Paul Gariy                                         #
#                                                            #
##############################################################
import sys
import argparse
import copy
import pickle
import random
import numpy as np
import os

import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, Dropout, InputLayer, Dense
from keras import Input, Model
from keras.utils import plot_model

import matplotlib.pyplot as plt
from matplotlib import colors

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


def build_model(
    n_inputs: int,
    n_hidden: int,
    n_output: int,
    activation: str = "elu",
    lrate: float = 0.001,
) -> Sequential:
    """
    Construct a network with n>=1 number of hidden layers
    - Adam optimizer
    - MSE loss

    :param n_inputs: Number of input dimensions
    :param n_hidden: Number of units in each of the hidden layer
    :param n_output: Number of ouptut dimensions
    :param activation: Activation function to be used for hidden and output units
    :param lrate: Learning rate for Adam Optimizer

    """
    model = Sequential()
    model.add(InputLayer(shape=(n_inputs,)))

    # Hidden layers
    for i, n in enumerate(n_hidden):
        model.add(Dense(n, use_bias=True, name="hidden_%d" % i, activation=activation))

    model.add(Dense(n_output, use_bias=True, name="output", activation=activation))

    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)

    # Bind the optimizer and the loss function to the model
    model.compile(optimizer=opt, loss="mse")

    # Generate an ASCII representation of the architecture
    if args.verbose >= 1:
        print(model.summary())

    return model


def args2string(args: argparse.ArgumentParser) -> str:
    """
    Translate the current set of arguments

    :param args: Command line arguments
    """
    return "exp_%02d_hidden_%s" % (args.exp, "_".join([str(i) for i in args.hidden]))


def execute_experiment(args: argparse.ArgumentParser):
    """
    Execute a single instance of an experiment.  The details are specified in the args object

    :param args: Command line arguments
    """

    # initialise wandb reporting
    wandb.init(project="Simple Test WandB", config=dict(vars(args)))
    # Read in the training set for function
    with open(args.dtpath, "rb") as f:
        data = pickle.load(f)
        ins = np.array(data["ins"])
        outs = np.array(data["outs"])

    # Create the model
    model = build_model(
        ins.shape[1], args.hidden, outs.shape[1], args.activation, args.lr
    )

    #####Callbacks
    # Early Stopping Callback
    es_cb = keras.callbacks.EarlyStopping(
        patience=10000,
        restore_best_weights=True,
        min_delta=0.001,
        monitor="loss",
        start_from_epoch=1000,
    )

    # Describe arguments
    argstring = args2string(args)
    print("EXPERIMENT: %s" % argstring)

    # Output pickle file
    fname_output = "results/xor_results_%s.pkl" % (argstring)

    # If file exists, skip the experiment
    if os.path.exists(fname_output):
        print("File %s already exists.Skipping experiment %s"% (fname_output, args.exp))
        return

    # Only execute if we are 'going'
    if not args.nogo:
        # Training
        print("Training...")

        # Fit the model
        history = model.fit(
            x=ins,
            y=outs,
            epochs=args.epochs,
            verbose=args.verbose >= 2,
            callbacks=[
                es_cb,
                WandbMetricsLogger(log_freq=5),
                WandbModelCheckpoint("models/models.keras"),
            ],
        )

        print("Done Training")
        

        #compute predictions on the training set
        predictions = model.predict(ins)
        #calculare absolute error for each training example
        abs_error = np.abs(predictions - outs)
        max_error = float(np.max(abs_error))
        sum_error = float(np.sum(abs_error))
        count_above_0_1 = int(np.sum(abs_error > 0.1))

        #log metrics to WandB
        wandb.log(
            {
                "max_error": max_error,
                "sum_error": sum_error,
                "count_above_0.1": count_above_0_1,
            }
        )

        #training metrics
        metrics = {
            "max_error": max_error,
            "sum_error": sum_error,
            "count_above_0.1": count_above_0_1,
        }
   
        # Save the training history
        with open(fname_output, "wb") as fp:
            pickle.dump(history.history, fp)
            pickle.dump(metrics, fp)
            pickle.dump(args, fp)
        fp.write
        wandb.finish()
     

def display_learning_curve(fname: str):
    """
    Display the learning curve that is stored in fname.
    As written, effectively assumes local execution
    (but could write the image out to a file)

    :param fname: Results file to load and dipslay

    """

    # Load the history file and display it
    with open(fname, "rb") as fp:
        history = pickle.load(fp)

    fp.close()

    # Display
    plt.plot(history["loss"])
    plt.ylabel("MSE")
    plt.xlabel("epochs")
    plt.title(fname)
    plt.savefig(fname.replace(".pkl", ".png"))


def create_parser() -> argparse.ArgumentParser:
    """
    Create a command line parser object for experiment
    """

    parser = argparse.ArgumentParser(description="Function Learner")

    parser.add_argument("--exp", type=int, default=0, help="Experimet Number")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument(
        "--hidden", nargs="+", type=int, default=[2], help="Number of hidden units"
    )

    parser.add_argument(
        "--activation", type=str, default="sigmoid", help="Activation function"
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Verbosity level"
    )
    parser.add_argument(
        "--dtpath", type=str, default="data.pkl", help="Path to data file"
    )
    parser.add_argument(
        "--nogo", action="store_true", help="Do not perform the experiment"
    )
    return parser


if __name__ == "__main__":
    # parse command line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Execute the experiment
    execute_experiment(args)
