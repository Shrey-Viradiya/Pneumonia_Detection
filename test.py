import os
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
    test_data, batch_size=batch_size, shuffle=False
)

# get all file names
filenames = os.listdir(test_data)

# testing the data
predictions = model.test(torch.nn.CrossEntropyLoss(), test_data_loader, device=device)
labels = ("Pneumonia", "Normal")
predictions = [ labels[p] for p in predictions ]
# print(f"Number of 0 predictions: {np.sum(predictions == 0)}")
# print(f"Number of 1 predictions: {np.sum(predictions == 1)}")
with open("test_predictions.csv", "w") as f:
    f.write("Image, Prediction\n")
    for i in range(len(predictions)):
        f.write(f"{filename[i]}, {predictions[i]}\n")

