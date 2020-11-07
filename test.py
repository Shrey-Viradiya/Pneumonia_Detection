import sys

import torch
import torchvision

from model import CoronaDetection, img_test_transforms

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

test_data_path = sys.argv[1]
test_data = torchvision.datasets.ImageFolder(
    root=test_data_path, transform=img_test_transforms
)
batch_size = 32
test_data_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True
)

model = CoronaDetection(sys.argv[2])
model.test(torch.nn.CrossEntropyLoss(), test_data_loader, device=device)
