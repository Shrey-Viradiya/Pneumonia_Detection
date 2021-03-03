# Pneumonia  Detection

This repository contains code for pneumonia detection using X-ray images of the lungs. Test accuracy achieved (highest) : 96.5 %

[Link for the dataset](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset)

[Link to the trained models](https://drive.google.com/drive/folders/1fx8SVrtOdtTdeOXox7aAZGe6RzjcW5Uk?usp=sharing)

## Instructions

- Install required packages from `requirements.txt`
- You can download, extract and move the data according to the label using data.py
    - `pip install kaggle`
    - Download Kaggle API key from your Kaggle Account. Go to `www.kaggle.com -> My Account -> Create New API token`
    - Place the file in on your home directory `$(HOME)/.kaggle/kaggle.json`.
    - Kaggle module will look for this token at `~/.kaggle/kaggle.json` on Linux, OSX, and other UNIX-based operating systems, and at `C:\Users\<windows-username>\.kaggle\kaggle.json` on Windows.
    - execute `python data.py`
- Run `main.py` to train from the dataset. For example, `python main.py --base_model {base_model} --optimizer {optimizer} --learning_rate {learning_rate} --batch_size {batch_size} --epoch {epoch} --nvidiadali --colab`
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
    - For faster training and Image augmentation, if on Linux, NVIDIA DALI can be used
        - Install NVIDIA DALI
        - Use --nvidiadali flag to use it
        - Note that NVIDIA DALI is only available for Linux
    - To change optimizers, use one of the following
        - Adam
        - SGD
        - RMSprop
        - Adagrad
        - Adadelta
- To test new dataset, run `test.py` with directory path and base model with option given above. Make sure that model is trained on those models first. 
    - Example, `python3 test.py "./data/Corona_Classification_data/test/" Inception True True`
- To generate a Class Activation Map from a trained model, after training it, use `CAM.py` like `python3 CAM.py {Model_Name} {Path_to_Image}`
    - Example, ```python3 CAM.py Inception "./data/abc.jpg"```
    - This will save the output as CAM_{Model_Name}.jpg in the current directory
