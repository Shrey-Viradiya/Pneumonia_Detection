import torch
import pickle
import os
import time
import torchvision

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class CoronaDetection():
    """
    Model Architecture and Forward Training Path for the Corona Detection

    Idea is to use transfer Learning
    """
    def __init__(self):

        if os.path.exists('./model/ConvModel'):
            self.model = torch.load('./model/ConvModel')
        else:
            self.model = torchvision.models.googlenet(pretrained=True)
            for name, param in self.model.named_parameters():
                if "bn" not in name:
                    param.requires_grad = False
            self.model.fc = torch.nn.Sequential(
                torch.nn.Linear(self.model.fc.in_features, 500),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(500, 2)
            )
            torch.save(self.model, './model/ConvModel')

    def train(self, optimizer, loss_fun, train_data ,test_data, epochs = 20, device = 'cuda'):
        '''
        Train function:
        parameters:
        optimizer   : optimizer object
        loss_fun    : Loss Function object
        train_data  : train dataloader
        test_data   : test  dataloader
        epochs      : default value 20
        device      : 'cuda' or 'cpu', default 'cuda'
        '''

        self.model.to(device)

        max_accurracy = 0.0

        for epoch in range(epochs):
            start = time.time()

            training_loss = 0.0
            valid_loss = 0.0

            self.model.train()
            correct = 0 
            total = 0
            for batch in train_data:
                train_images, train_labels = batch
                train_images = train_images.to(device)
                train_labels = train_labels.to(device)
                
                optimizer.zero_grad()
                output = self.model(train_images)
                loss = loss_fun(output, train_labels)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += train_labels.size(0)
                correct += (predicted == train_labels).sum().item()
            training_accuracy = correct/total * 100

            self.model.eval()
            correct = 0 
            total = 0
            with torch.no_grad():
                for batch in test_data:
                    test_images, test_labels = batch
                    test_images = test_images.to(device)
                    test_labels = test_labels.to(device)

                    output = self.model(test_images)
                    loss = loss_fun(output,test_labels) 
                    valid_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
            testing_accuracy = correct/total * 100

            if (testing_accuracy > max_accurracy):
                    max_accurracy = testing_accuracy
                    torch.save(self.model, './model/ConvModel')

            print(f'{bcolors.OKGREEN}Epoch:{bcolors.ENDC} {epoch + 1}, {bcolors.OKGREEN}Training Loss:{bcolors.ENDC} {training_loss:.5f}, {bcolors.OKGREEN}Validation Loss:{bcolors.ENDC} {valid_loss:.5f}, {bcolors.OKGREEN}Training accuracy:{bcolors.ENDC} {training_accuracy:.2f} %, {bcolors.OKGREEN}Testing accuracy:{bcolors.ENDC} {testing_accuracy:.2f} %, {bcolors.OKGREEN}time:{bcolors.ENDC} {time.time() - start:.2f} s')

    def test(self, loss_fun, test_data, device = 'cuda'):
        print("Starting Evaluating....")
        start = time.time()
        self.model.eval()
        test_loss = 0.0
        correct = 0 
        total = 0
        with torch.no_grad():
            for batch in test_data:
                test_images, test_labels = batch
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)

                output = self.model(test_images)
                loss = loss_fun(output,test_labels) 
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
        testing_accuracy = correct/total * 100

        print(f'{bcolors.OKGREEN}Test Loss:{bcolors.ENDC} {test_loss:.5f}, {bcolors.OKGREEN}Testing accuracy:{bcolors.ENDC} {testing_accuracy:.2f} %, {bcolors.OKGREEN}time:{bcolors.ENDC} {time.time() - start:.2f} s')
