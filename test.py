import os
import sys

import numpy as np
import torch
import torchvision

from model import PneumoniaDetection, pretrained_models
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from NvidiaDali import *

if len(sys.argv) != 5:
    print("usage: python test.py <data_directory_path> <model_name> <has_labels_flag> <NVIDIA-DALI>")
    print("<data_directory_path>: Path to directory containing the test data")
    print(
        "<model_name>: Name of the trained model. See 'pretrained_models' for details"
    )
    print("<has_labels_flag>: Whether the directory contains test labels or not.")
    print("<NVIDIA-DALI>: Whether to use nvidia-dali or not.")
    exit(1)

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

# testing the data
bool_dict = {
    "True": True,
    "False": False,
    "true": True,
    "false": False,
    "t": True,
    "f": False,
}

# creating model object
model = PneumoniaDetection(sys.argv[2])

dali_bool = bool_dict[sys.argv[4]]

batch_size = 8

if dali_bool:
    
    TEST_SIZE = 0     # set this variable manually, equals to total number of files in testing
    TEST_STEPS = TEST_SIZE // batch_size

    assert TEST_SIZE != 0
    pipe = HybridPipelineTest(batch_size=batch_size, output_size = list(pretrained_models[sys.argv[2]][2]), num_threads=2, device_id=0, images_directory=test_data_path)
    pipe.build()

    test_data_loader = DALIClassificationIterator([pipe], size=TEST_STEPS)
else:
    test_data_path = sys.argv[1]
    test_data = torchvision.datasets.ImageFolder(
        root=test_data_path, transform=model.test_transformation
    )
    
    test_data_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

# get all file names
dirname = os.listdir(test_data_path)
filename = []
for d in dirname:
    filename.extend(os.listdir(os.path.join(test_data_path, d)))

predictions = model.test(
    torch.nn.CrossEntropyLoss(),
    test_data_loader,
    device=device,
    has_labels=bool_dict[sys.argv[3]],
    dali = dali_bool
)
labels = ("Pneumonia", "Normal")
predictions = [labels[p] for p in predictions]

# make the prediction file
with open(f"test_predictions_{sys.argv[2]}.csv", "w") as f:
    f.write("Image, Prediction\n")
    for i in range(len(predictions)):
        f.write(f"{filename[i]}, {predictions[i]}\n")
