import torch
import torchvision
from model import *

img_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((256,256)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    ])

train_data_path = "./data/Corona_Classification_data/train/"
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms)

test_data_path = "./data/Corona_Classification_data/test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=img_transforms)

batch_size=32

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader  = torch.utils.data.DataLoader(test_data  , batch_size=batch_size, shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")

model = CoronaDetection()
optimizer = torch.optim.Adam(model.model.parameters(), lr=0.00005)

model.train(optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, test_data_loader, epochs=25, device=device)