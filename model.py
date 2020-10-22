import torch
import pickle
import os
import time
import torchvision

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class CoronaDetection():
    """
    Model Architecture and Forward Training Path for the Corona Detection

    Idea is to use transfer Learning
    """
    def __init__(self, data_directory):

        self.data_directory = data_directory

        if os.path.exists('./model/ConvModel'):
            self.model = torch.load('./model/ConvModel')
        else:
            self.model = torchvision.models.googlenet(pretrained=True)
            for name, param in self.model.named_parameters():
                if "bn" not in name:
                    param.requires_grad = False
            self.model.fc = torch.nn.Sequential(
                torch.nn.Linear(self.model.fc.in_features, 500),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(500, 2)
            )
            torch.save('./model/ConvModel')