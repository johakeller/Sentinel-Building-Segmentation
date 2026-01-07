"""Module to start the entire programm or single partes of it."""

import os
import sys
import torch

import params
import data_acquisition
import train_apply


def check_data():
    """
    Confirms before training, whether image data ( .pkl files) already exists.

        Args:
            None

        Returns:
            bool: Image data is there or not
    """

    # check if data is already there
    for city in params.CITIES:
        if not os.path.exists(os.path.join(params.IMAGE_DATA_PATH, f"{city}.pkl")):
            # check also for dataset
            if not os.path.exists(
                os.path.join(params.DATASET_TRAIN, f"{city}_data.mmap")
            ):
                return False
    if not os.path.exists(
        os.path.join(params.IMAGE_DATA_PATH, f"{params.TEST_CITY}.pkl")
    ):
        # check also for dataset
        if not os.path.exists(
            os.path.join(params.DATASET_TEST, f"{params.TEST_CITY}_data.mmap")
        ):
            return False
    # everything there
    return True


def main(args):
    """
    Main function to start the program with given arguments.

    Args:
        args (list): passed arguments

    Returns:
        None
    """

    # run entire pipeline if no argument passed
    if len(args) == 0:
        print(
            "Run entire pipeline: acquisition, train of UNet, training of ConvNet, data augmentation with UNet, data augmentation with ConvNet"
        )
        data_acquisition.run_acquisition(plot=False)
        train_apply.train_apply_hyper("UNet")
        train_apply.train_apply_hyper("ConvNet")
        train_apply.augment_apply("UNet")
        train_apply.augment_apply("ConvNet")

    # run acquisition with or without plotting
    if "acq" in args:
        if "plot" in args:
            print("Run data acquisition with plotting.")
            data_acquisition.run_acquisition(plot=True)
        else:
            print("Run data acquisition.")
            data_acquisition.run_acquisition(plot=False)

    if check_data():
        # UNet
        if "unet" in args:
            if (
                args.index("unet") + 1 < len(args)
                and args[args.index("unet") + 1] == "augment"
            ):
                # augmentation train test
                print("Run data augmentation with UNet.")
                train_apply.augment_apply("UNet")
            elif (
                args.index("unet") + 1 < len(args)
                and args[args.index("unet") + 1] == "hyper"
            ):
                # normal hyperparameter optimization
                print("Run hyperparamter optimization with UNet.")
                train_apply.train_apply_hyper("UNet")
            else:
                # no hyperparameter optimization
                print("Run simple training, validation with UNet.")
                train_apply.train_apply("UNet")

        # ConvNet
        if "convnet" in args:
            if (
                args.index("convnet") + 1 < len(args)
                and args[args.index("convnet") + 1] == "augment"
            ):
                # augmentation train test
                print("Run data augmentation with ConvNet.")
                train_apply.augment_apply("ConvNet")
            elif (
                args.index("convnet") + 1 < len(args)
                and args[args.index("convnet") + 1] == "hyper"
            ):
                # normal hyperparameter optimization
                print("Run hyperparamter optimization with ConvNet.")
                train_apply.train_apply_hyper("ConvNet")
            else:
                # no hyperparameter optimization
                print("Run simple training, validation with ConvNet.")
                train_apply.train_apply("ConvNet")
    else:
        print(
            "Image data is not available, please run acquisition first: $ python main.py <acq>"
        )
        sys.exit(1)


if __name__ == "__main__":
    # passed system arguments
    given_args = sys.argv[1:]

    # valid arguments
    val_args = ["acq", "plot", "unet", "convnet", "augment", "hyper"]

    # check arguments are valid
    if len(given_args) != 0:
        for arg in given_args:
            if arg not in val_args:
                print(
                    "Usage: $ python main.py [acq] [plot]/([unet] [hyper]/[augment])/[convnet] [augment]"
                )
                sys.exit(1)

    # against threading bug
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.multiprocessing.set_start_method("spawn")

    main(given_args)
