import os
import sys

import numpy as np
import torch
import torchvision

from model import CoronaDetection

if len(sys.argv) != 4:
    print("usage: python test.py <data_directory_path> <model_name> <has_labels_flag>")
    print("<data_directory_path>: Path to directory containing the test data")
    print(
        "<model_name>: Name of the trained model. See 'pretrained_models' for details"
    )
    print("<has_labels_flag>: Whether the directory contains test labels or not.")
    exit(1)

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
dirname = os.listdir(test_data_path)
filename = []
for d in dirname:
    filename.extend(os.listdir(os.path.join(test_data_path, d)))

# testing the data
bool_dict = {
    "True": True,
    "False": False,
    "true": True,
    "false": False,
    "t": True,
    "f": False,
}
predictions = model.test(
    torch.nn.CrossEntropyLoss(),
    test_data_loader,
    device=device,
    has_labels=bool_dict[sys.argv[3]],
)
labels = ("Pneumonia", "Normal")
predictions = [labels[p] for p in predictions]

# make the prediction file
with open(f"test_predictions_{sys.argv[2]}.csv", "w") as f:
    f.write("Image, Prediction\n")
    for i in range(len(predictions)):
        f.write(f"{filename[i]}, {predictions[i]}\n")
