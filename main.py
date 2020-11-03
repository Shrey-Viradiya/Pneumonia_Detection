import torch
import torchvision
from model import *
import sys

print("Corona Detection Project")

print("Setting up Data Directories")
train_data_path = "./data/Corona_Classification_data/train/"
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_train_transforms)

test_data_path = "./data/Corona_Classification_data/test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=img_test_transforms)

batch_size=4

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader  = torch.utils.data.DataLoader(test_data  , batch_size=batch_size, shuffle=True)

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda") 
else:
    print("Using CPU")
    device = torch.device("cpu")

print("Creating Model Object: ")
model = CoronaDetection(sys.argv[1])
optimizer = torch.optim.Adam(model.model.parameters(), lr=0.00005)

print("Starting Training")
model.train(optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, test_data_loader, epochs=25, device=device)
print("Completed Training")

model.CAM('./data/Corona_Classification_data/train/INFECTED/4C4DEFD8-F55D-4588-AAD6-C59017F55966.jpeg', f'Overlay_{sys.argv[1]}.jpg')
