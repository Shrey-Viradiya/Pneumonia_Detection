import torch
import torchvision
from model import *
import sys

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda") 
else:
    print("Using CPU")
    device = torch.device("cpu")

model = CoronaDetection()

img_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((64,64)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    ])

test_data_path = sys.argv[1]
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=img_transforms)
batch_size=32
test_data_loader  = torch.utils.data.DataLoader(test_data  , batch_size=batch_size, shuffle=True)

model.test(torch.nn.CrossEntropyLoss(), test_data_loader, device = device)