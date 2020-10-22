import os
import sys

if not os.path.exists('./data/coronahack-chest-xraydataset.zip'):
    os.system('kaggle datasets download -d praveengovi/coronahack-chest-xraydataset')
else:
    print("Download found....")

print("Strating Extraction")
os.system('tar -xf "./data/coronahack-chest-xraydataset.zip" --directory "./data/" ')
print("Extraction Complete")