import sys

import numpy as np
import torch
import torchvision

from model import CoronaDetection

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

# creating model object
model = CoronaDetection(sys.argv[2])

test_data_path = sys.argv[1]
test_data = torchvision.datasets.ImageFolder(
    root=test_data_path, transform=model.test_transformation
)
batch_size = 8
test_data_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True
)

# testing the data
predictions = model.test(torch.nn.CrossEntropyLoss(), test_data_loader, device=device)
predictions = np.asarray(predictions)
print(f"Number of 0 predictions: {np.sum(predictions == 0)}")
print(f"Number of 1 predictions: {np.sum(predictions == 1)}")