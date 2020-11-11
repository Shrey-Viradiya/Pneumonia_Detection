# Corona Detection

This repository contains code for corona detection using X-ray images of the lungs.

Link for the dataset is https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset

## Instructions

- Install required packages from requirements.txt

- You can download, extract and move the data according to the label using data.py
    - pip install kaggle
    - Download Kaggle API key from your Kaggle Account. Homepage www.kaggle.com -> My Account -> Create New API token
    - Place it in the appropriate place according to the OS.
    - kaggle module will look for this token at ```~/.kaggle/kaggle.json``` on Linux, OSX, and other UNIX-based operating systems, and at ```C:\Users\<Windows-username>\.kaggle\kaggle.json``` on Windows.
    - execute ```python data.py```

- run ```main.py``` to train from the dataset like ```python3 main.py {Model_Name} {learning_rate} {batch_size} {epochs}```

    Example, ```python3 main.py Inception 0.0001 32 20```

    - To change pretrained base model, give input while initializing the model object and in command line argument.  Use values from 
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
        - DenseNet161
        - GoogleNet
        - Inception

- To test new dataset, run ```test.py``` with directory path and base model with option given above. Make sure that model is trained on those models first. 

    Example, ```python3 test.py "./data/Corona_Classification_data/test/" Inception``` 
    
    
- To generate a Class Activation Map from a trained model, after training it, use ```CAM.py``` like ```python3 CAM.py {Model_Name} {Path_to_Image}```
    
    Example, ```python3 CAM.py Inception "./data/abc.jpg"```
    
    this will save the output as CAM_{Model_Name}.jpg in the current directory
