import os
import sys
import time
from PIL import Image

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import skimage.transform


# Dictionary for pretrained models and their last layer name
pretrained_models = {
    "ResNet18": [torchvision.models.resnet18, "layer4"],
    "ResNet34": [torchvision.models.resnet34, "layer4"],
    "ResNet50": [torchvision.models.resnet50, "layer4"],
    "ResNet101": [torchvision.models.resnet101, "layer4"],
    "ResNet152": [torchvision.models.resnet152, "layer4"],
    "Alexnet": [torchvision.models.alexnet, "features"],
    "VGG11": [torchvision.models.vgg11_bn, "features"],
    "VGG13": [torchvision.models.vgg13_bn, "features"],
    "VGG16": [torchvision.models.vgg16_bn, "features"],
    "VGG19": [torchvision.models.vgg19_bn, "features"],
    "GoogleNet": [torchvision.models.googlenet, "inception5b"],
    "Inception": [torchvision.models.inception_v3, "Mixed_7c"],
}


# Different image transformations for training, testing and displaying
img_train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomPerspective(),
        torchvision.transforms.RandomRotation(25),
        torchvision.transforms.RandomResizedCrop(
            (512, 512),
            scale=(0.7, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=2,
        ),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

display_transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((512, 512))]
)

img_test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


# The main model object
class CoronaDetection:
    """
    Model Architecture and Forward Training Path for the Corona Detection
    Idea is to use transfer Learning
    """

    def __init__(self, base_model="ResNet18"):
        assert base_model in [
            "ResNet18",
            "ResNet34",
            "ResNet50",
            "ResNet101",
            "ResNet152",
            "Alexnet",
            "VGG11",
            "VGG13",
            "VGG16",
            "VGG19",
            "GoogleNet",
            "Inception",
        ]

        # saving base model name to use it in saving the model
        self.base_model = base_model

        if os.path.exists(f"./model/ConvModel_{self.base_model}"):
            # check if the model is intialized before
            self.model = torch.load(f"./model/ConvModel_{self.base_model}")
            # print(self.model)
        else:
            # If not initialized before
            # Download it and save it
            self.model = pretrained_models[self.base_model][0](pretrained=True)
            for name, param in self.model.named_parameters():
                param.requires_grad = True

            # Modify last Fully Connected layer to predict for
            # Our requirements
            if self.base_model in ["Alexnet", "VGG16"]:
                num_ftrs = self.model.classifier[6].in_features
                self.model.classifier[6] = torch.nn.Linear(num_ftrs, 2)
            else:
                self.model.fc = torch.nn.Sequential(
                    torch.nn.Linear(self.model.fc.in_features, 500),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(),
                    torch.nn.Linear(500, 2),
                )

            # Save model
            torch.save(self.model, f"./model/ConvModel_{self.base_model}")

        # get final model for using it in Class Activation Map
        self.final_layer = self.model._modules.get(
            pretrained_models[self.base_model][1]
        )

    def train(
        self,
        optimizer,
        loss_fun,
        train_data,
        test_data,
        epochs=20,
        early_stopping_threshold=4,
        device="cuda",
    ):
        """
        Train function:
        parameters:
        optimizer   : optimizer object
        loss_fun    : Loss Function object
        train_data  : train dataloader
        test_data   : test  dataloader
        epochs      : default value 20
        early_stopping_threshold : Early stopping threshold
        device      : 'cuda' or 'cpu', default 'cuda'
        """

        # transfer model to device available
        self.model.to(device)

        max_accurracy = 0.0
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        for epoch in range(epochs):
            start = time.time()

            training_loss = 0.0
            valid_loss = 0.0

            train_correct = 0
            train_total = 0

            # Training over batches
            self.model.train(mode=True)
            for train_batch in train_data:
                train_images, train_labels = train_batch
                train_images = train_images.to(device)
                train_labels = train_labels.to(device)

                optimizer.zero_grad()
                train_output = self.model(train_images)

                if self.base_model == "Inception":
                    train_output = train_output.logits

                train_loss = loss_fun(train_output, train_labels)

                train_loss.backward()
                optimizer.step()

                training_loss += train_loss.item()

                _, train_predicted = torch.max(train_output.data, 1)

                train_total += train_labels.size(0)
                train_ccount = (train_predicted == train_labels).sum().item()
                train_correct += train_ccount

                sys.stdout.write(
                    f"\rEpoch {epoch+1}\t"
                    f"Training Loss => {train_loss:.4f}\t"
                    f"Training Acc => "
                    f"{train_ccount/train_images.shape[0]*100:5.2f}"
                )

            training_accuracy = train_correct / train_total * 100

            valid_loss = 0.0
            test_correct = 0
            test_total = 0
            misses = 0
            previous_accuracy = 0

            # Test over batches
            self.model.train(mode=False)
            with torch.no_grad():
                for test_batch in test_data:
                    test_images, test_labels = test_batch
                    test_images = test_images.to(device)
                    test_labels = test_labels.to(device)

                    test_output = self.model(test_images)

                    if self.base_model == "Inception":
                        test_output = test_output.logits

                    test_loss = loss_fun(test_output, test_labels)
                    valid_loss += test_loss.item()

                    _, test_predicted = torch.max(test_output.data, 1)

                    test_total += test_labels.size(0)
                    test_ccount = (test_predicted == test_labels).sum().item()
                    test_correct += test_ccount

            testing_accuracy = test_correct / test_total * 100

            sys.stdout.flush()
            sys.stdout.write("\r")

            time_taken = time.time() - start

            print(
                f"Epoch {epoch + 1}\t"
                f"Training Loss => {training_loss:.4f}\t"
                f"Training Acc => {training_accuracy:5.2f}\t"
                f"Test Loss => {valid_loss:.4f}\t"
                f"Test Acc => {testing_accuracy:5.2f}\t"
                f"Time Taken => {time_taken:5.2f}"
            )

            train_losses.append(training_loss)
            test_losses.append(valid_loss)
            train_accuracies.append(training_accuracy)
            test_accuracies.append(testing_accuracy)

            # Save if it is better model than max_accuracy
            if testing_accuracy > max_accurracy:
                max_accurracy = testing_accuracy
                torch.save(self.model, f"./model/ConvModel_{self.base_model}")

                with open(
                    f"./model/ConvModel_{self.base_model}_results.txt", "w"
                ) as f:
                    f.writelines(
                        [
                            f"BaseModel: {self.base_model}\n",
                            f"Epochs: {epoch + 1}\n",
                            f"Train Dataloader Batch Size: {train_data.batch_size}\n",
                            f"Test Dataloader Batch Size: {test_data.batch_size}\n",
                            f'Params for Adam: {optimizer.__dict__["defaults"]}\n',
                            f"Training Loss: {training_loss}\n",
                            f"Validation Loss: {valid_loss}\n",
                            f"Training Accuracy: {training_accuracy}\n",
                            f"Testing Accuracy: {testing_accuracy}\n",
                            f"Time Taken: {time_taken} seconds",
                        ]
                    )

            # Decide and stop early if needed
            if epoch >= 1:
                if (
                    previous_accuracy > testing_accuracy
                    and misses < early_stopping_threshold
                ):
                    misses += 1
                    previous_accuracy = testing_accuracy
                elif previous_accuracy > testing_accuracy:
                    print(f"Early Stopping....")
                    print(
                        f"Epoch {epoch + 1}\t"
                        f"Training Loss => {training_loss:.4f}\t"
                        f"Training accuracy => {training_accuracy:5.2f}\t"
                        f"Test Loss => {valid_loss:.4f}\t"
                        f"Testing accuracy => {testing_accuracy:5.2f}\t"
                        f"Time Taken => {time_taken:.2f}"
                    )
                    break
            previous_accuracy = testing_accuracy

            np.save(f"./model/train_losses_{self.model.base_model}", train_losses)
            np.save(f"./model/train_accuracies_{self.model.base_model}", train_accuracies)
            np.save(f"./model/test_losses_{self.model.base_model}", test_losses)
            np.save(f"./model/test_accuracies_{self.model.base_model}", test_accuracies)

    def test(self, loss_fun, test_data, device="cuda"):
        print("Starting Evaluating....")
        start = time.time()
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        # Without changing parameters
        with torch.no_grad():
            # Testing over batches
            for batch in test_data:
                test_images, test_labels = batch
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)

                output = self.model(test_images)
                loss = loss_fun(output, test_labels)
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
        testing_accuracy = correct / total * 100

        print(
            f"Test Loss => {test_loss:.5f}\t"
            f"Testing accuracy => {testing_accuracy:.2f}\t"
            f"Time Taken => {time.time() - start:.2f}"
        )

    def CAM(self, image_path_input, overlay_path_output, device="cuda"):
        """
        CAM - Class Activation Map
        """

        # open image
        image = Image.open(image_path_input)
        image = image.convert('RGB')
        print(image.mode)

        tensor = img_test_transforms(image)

        prediction_var = torch.autograd.Variable(
            (tensor.unsqueeze(0)).cuda(), requires_grad=True
        )
        self.model.to(device)
        self.model.eval()

        class SaveFeatures:
            features = None

            def __init__(self, m):
                self.hook = m.register_forward_hook(self.hook_fn)

            def hook_fn(self, module, input, output):
                self.features = ((output.cpu()).data).numpy()

            def remove(self):
                self.hook.remove()

        activated_features = SaveFeatures(self.final_layer)
        prediction_var = prediction_var.to(device)
        prediction = self.model(prediction_var)

        pred_probabilities = torch.nn.functional.softmax(
            prediction, dim=0
        ).data.squeeze()

        activated_features.remove()

        torch.topk(pred_probabilities, 1)

        def getCAM(feature_conv, weight_fc, class_idx):
            _, nc, h, w = feature_conv.shape
            cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            return [cam_img]

        weight_softmax_params = list(self.model._modules.get("fc").parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

        class_idx = torch.topk(pred_probabilities, 1)[1].int()

        overlay = getCAM(activated_features.features, weight_softmax, class_idx)

        plt.figure(figsize=(32,15))
        plt.subplot(1, 2, 1)
        plt.imshow(display_transform(image))
        plt.subplot(1, 2, 2)
        plt.imshow(display_transform(image))
        plt.imshow(
            skimage.transform.resize(overlay[0], tensor.shape[1:3]),
            alpha=0.4,
            cmap="jet",
        )
        plt.savefig(overlay_path_output)
