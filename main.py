import torch
import torchvision
import argparse
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from NvidiaDali import *

from model import PneumoniaDetection, pretrained_models

DESCRIPTION = """
Train the Corona Detection Model
"""

USAGE = """py runner.py --base_model base_model 

Run Corona Detection Model on your machine.

For example, you can pass the following parameters:

python main.py --base_model Inception --optimizer Adam --learning_rate 0.00001 --batch_size 32 --epoch 25 --colab

This will run main.py with base model as Inception and given parameters

"""

parser = argparse.ArgumentParser(description=DESCRIPTION, usage=USAGE)
parser.add_argument(
    "--base_model",
    metavar="base_model",
    type=str,
    action="store",
    help="Base Model to use. Available Options: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, Alexnet, VGG11, VGG13, VGG16, VGG19, GoogleNet, Inception",
    required=True,
)
parser.add_argument(
    "--optimizer",
    metavar="optimizer",
    type=str,
    action="store",
    help="Optimizer to use in the training",
    required=True,
)
parser.add_argument(
    "--learning_rate",
    metavar="learning_rate",
    type=float,
    action="store",
    help="Learning Rate for training",
    default=0.00001,
)
parser.add_argument(
    "--batch_size",
    metavar="batch_size",
    type=int,
    action="store",
    help="Batch_size",
    default=16,
)
parser.add_argument(
    "--epoch", metavar="epoch", type=int, action="store", help="Epoch", default=15
)

parser.add_argument(
    "--nvidiadali",
    action="store_true",
    help="Option to use when want to use nvidiadali...need to install it before running",
)

parser.add_argument(
    "--colab",
    action="store_true",
    help="Option to use when using colab for training...Mount the drive and it will be saved in the drive",
)

optimizers = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop,
    "Adagrad": torch.optim.Adagrad,
    "Adadelta": torch.optim.Adadelta,
}

if __name__ == "__main__":
    kwargs = vars(parser.parse_args())

    print("Corona Detection Project")

    if torch.cuda.is_available():
        print("Using GPU")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    print("Creating Model Object: ")
    model = PneumoniaDetection(kwargs["base_model"], colab=kwargs["colab"])

    train_data_path = "./data/Corona_Classification_data/train/"
    test_data_path = "./data/Corona_Classification_data/test/"
    batch_size = kwargs["batch_size"]
    learning_rate = kwargs["learning_rate"]

    if kwargs["nvidiadali"]:
        pipe = HybridPipelineTrain(batch_size=batch_size, output_size = list(pretrained_models[kwargs["base_model"]][2]), num_threads=2, device_id=0, images_directory=train_data_path)
        pipe.build()

        DATA_SIZE = 0   # set this variable manually, equals to total number of files in training
        VALIDATION_SIZE = 0     # # set this variable manually, equals to total number of files in training
        ITERATIONS_PER_EPOCH = DATA_SIZE // batch_size
        VALIDATION_STEPS = VALIDATION_SIZE // batch_size

        assert DATA_SIZE != 0
        assert VALIDATION_SIZE != 0

        dali_iter = DALIClassificationIterator([pipe], size=ITERATIONS_PER_EPOCH)

        pipe2 = HybridPipelineTest(batch_size=batch_size, output_size = list(pretrained_models[kwargs["base_model"]][2]), num_threads=2, device_id=0, images_directory=test_data_path)
        pipe2.build()

        valid_iter = DALIClassificationIterator([pipe2], size=VALIDATION_STEPS)
    else:
        print("Setting up Data Directories")
        
        train_data = torchvision.datasets.ImageFolder(
            root=train_data_path, transform=model.train_transformation
        )
        
        test_data = torchvision.datasets.ImageFolder(
            root=test_data_path, transform=model.test_transformation
        )    

        train_data_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        test_data_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True
        )

    

    optimizer = optimizers[kwargs["optimizer"]](
        model.model.parameters(), lr=learning_rate
    )

    print("Starting Training")

    if kwargs["nvidiadali"]:
        model.train(
            optimizer,
            torch.nn.CrossEntropyLoss(),
            dali_iter,
            valid_iter,
            epochs=kwargs["epoch"],
            device=device,
            dali=True
        )
    else:
        model.train(
            optimizer,
            torch.nn.CrossEntropyLoss(),
            train_data_loader,
            test_data_loader,
            epochs=kwargs["epoch"],
            device=device,
            dali=False
        )
    print("Completed Training")
