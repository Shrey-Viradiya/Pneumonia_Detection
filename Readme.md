# Corona Detection

This repository contains code for corona detection using X-ray images of the lungs.

Link for the dataset is https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset

## Instructions

- Install required packages from requirements.txt

- You can download, extract and move the data according to the label using data.py

    - Download Kaggle API key from you Kaggle Account
    - Place it in the appropriate place according to the OS.
    - kaggle module will look for this token at ```~/.kaggle/kaggle.json``` on Linux, OSX, and other UNIX-based operating systems, and at ```C:\Users\<Windows-username>\.kaggle\kaggle.json``` on Windows.
    - execute ```python data.py```

- run main.py to train from the dataset