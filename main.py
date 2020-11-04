import torch
import torchvision
from model import *
import sys

print("Corona Detection Project")

print("Setting up Data Directories")
train_data_path = "./data/Corona_Classification_data/train/"
train_data = torchvision.datasets.ImageFolder(
    root=train_data_path, transform=img_train_transforms
)

test_data_path = "./data/Corona_Classification_data/test/"
test_data = torchvision.datasets.ImageFolder(
    root=test_data_path, transform=img_test_transforms
)

batch_size = int(sys.argv[3])

train_data_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
test_data_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True
)

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

print("Creating Model Object: ")
model = CoronaDetection(sys.argv[1])

learning_rate = float(sys.argv[2])

optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)

print("Starting Training")
train_losses, train_accuracies, test_losses, test_accuracies = model.train(
    optimizer,
    torch.nn.CrossEntropyLoss(),
    train_data_loader,
    test_data_loader,
    epochs=25,
    device=device,
)
print("Completed Training")

train_losses, train_accuracies, test_losses, test_accuracies = map(
    np.asarray, [train_losses, train_accuracies, test_losses, test_accuracies]
)

np.save(f"./model/train_losses_{model.base_model}", train_losses)
np.save(f"./model/train_accuracies_{model.base_model}", train_accuracies)
np.save(f"./model/test_losses_{model.base_model}", test_losses)
np.save(f"./model/test_accuracies_{model.base_model}", test_accuracies)
