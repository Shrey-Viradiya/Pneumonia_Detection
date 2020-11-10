import sys
import torch
from model import CoronaDetection

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")

print("Creating Model Object: ")
model = CoronaDetection(sys.argv[1])

model.CAM('./data/Corona_Classification_data/test/INFECTED/person3_virus_17.jpeg',f'Infected_{sys.argv[1]}.jpg', device)