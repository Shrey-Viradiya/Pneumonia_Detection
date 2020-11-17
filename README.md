# Pneumonia  Detection

This repository contains code for pneumonia detection using X-ray images of the lungs.

Link for the dataset is https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset

## Instructions

- Install required packages from requirements.txt

- You can download, extract and move the data according to the label using data.py
    - pip install kaggle
    - Download Kaggle API key from your Kaggle Account. Homepage www.kaggle.com -> My Account -> Create New API token
    - Place it in the appropriate place according to the OS.
    - kaggle module will look for this token at ```~/.kaggle/kaggle.json``` on Linux, OSX, and other UNIX-based operating systems, and at ```C:\Users\<Windows-username>\.kaggle\kaggle.json``` on Windows.
    - execute ```python data.py```

- run ```main.py``` to train from the dataset like ```python main.py --base_model {base_model} --optimizer {optimizer} --learning_rate {learning_rate} --batch_size {batch_size} --epoch {epoch} --colab```

    Example, ```python main.py --base_model Inception --optimizer Adam --learning_rate 0.00001 --batch_size 32 --epoch 25 --colab```

    - To change pretrained base model, give input while initializing the model object. Use values from 
        - ResNet18 
        - ResNet34
        - ResNet50
        - ResNet101
        - ResNet152
        - Alexnet 
        - VGG11
        - VGG13
        - VGG16
        - VGG19
        - GoogleNet
        - Inception

    - If using colab for training, mount the drive and use --colab to save the files in the drive

    - To change optimizers, use one of the following
        - Adam
        - SGD
        - RMSprop
        - Adagrad
        - Adadelta

- To test new dataset, run ```test.py``` with directory path and base model with option given above. Make sure that model is trained on those models first. 

    Example, ```python3 test.py "./data/Corona_Classification_data/test/" Inception``` 
    
    
- To generate a Class Activation Map from a trained model, after training it, use ```CAM.py``` like ```python3 CAM.py {Model_Name} {Path_to_Image}```
    
    Example, ```python3 CAM.py Inception "./data/abc.jpg"```
    
    this will save the output as CAM_{Model_Name}.jpg in the current directory
