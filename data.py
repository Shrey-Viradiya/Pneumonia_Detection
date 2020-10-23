import os
import shutil
import pandas as pd

if not os.path.exists('./data/coronahack-chest-xraydataset.zip'):
    os.system('kaggle datasets download -d praveengovi/coronahack-chest-xraydataset')
else:
    print("Download found....")

print("Strating Extraction")
os.system('tar -xf "./data/coronahack-chest-xraydataset.zip" --directory "./data/" ')
print("Extraction Complete")

print("Reading Metadata")
images_data = pd.read_csv('./data/Chest_xray_Corona_Metadata.csv')

os.mkdir("./data/Corona_Classification_data")
os.mkdir("./data/Corona_Classification_data/train")
os.mkdir('./data/Corona_Classification_data/train/INFECTED')
os.mkdir("./data/Corona_Classification_data/train/NORMAL")
os.mkdir("./data/Corona_Classification_data/test")
os.mkdir('./data/Corona_Classification_data/test/INFECTED')
os.mkdir("./data/Corona_Classification_data/test/NORMAL")

print("Moving According to Labels")
for index, row in images_data.iterrows():
    if row['Label'] == 'Normal' and row['Dataset_type'] == 'TRAIN':
        shutil.move(f"./data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/{row['X_ray_image_name']}", f"./data/Corona_Classification_data/train/NORMAL/{row['X_ray_image_name']}")
        
    if row['Label'] == 'Pnemonia' and row['Dataset_type'] == 'TRAIN':
        shutil.move(f"./data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/{row['X_ray_image_name']}", f"./data/Corona_Classification_data/train/INFECTED/{row['X_ray_image_name']}")

    if row['Label'] == 'Normal' and row['Dataset_type'] == 'TEST':
        shutil.move(f"./data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/{row['X_ray_image_name']}", f"./data/Corona_Classification_data/test/NORMAL/{row['X_ray_image_name']}")

    if row['Label'] == 'Pnemonia' and row['Dataset_type'] == 'TEST':
        shutil.move(f"./data/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/{row['X_ray_image_name']}", f"./data/Corona_Classification_data/test/INFECTED/{row['X_ray_image_name']}")

print("Moving Complete")
shutil.rmtree('./data/Coronahack-Chest-XRay-Dataset')
print("Cleaning Complete")