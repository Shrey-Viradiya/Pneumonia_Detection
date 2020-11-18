# go to the parent directory on google colab
cd /content/

# Set up some flags to verify later...
HOMEDIR=`ls ~ -a`
FLAGKAGGLEDIR=1
FLAGKAGGLE=0

# check for existing files in the directory.
for FILENAME in $HOMEDIR
do
    if [ $FILENAME == ".kaggle" ]
    then
        # The .kaggle directory is present.
        FLAGKAGGLEDIR=0
        break
    fi
done

# If the '.kaggle' directory is not present, create one.
if [ $FLAGKAGGLEDIR -eq 1 ]
then
    mkdir ~/.kaggle
else
    # otherwise, mark the API key as present.
    FLAGKAGGLE=1
fi

# search this directory for other files and directories.
CURRDIR=`ls -a`
FLAGMASTER=0
FLAGDATA=0
FLAGMODEL=0
SETUP_DONE=0

for FILENAME in $CURRDIR
do
    # check if master is present
    if [ $FILENAME == "Pneumonia_Detection-master.zip" ]
    then
        FLAGMASTER=1
    # check if data directory is present
    elif [ $FILENAME == "data" ]
    then
        FLAGDATA=1
    # check if model is present
    elif [ $FILENAME == "model" ]
    then
        FLAGMODEL=1
    # check if new API key is present
    # if it is, we use it to rewrite the old key
    elif [ $FILENAME == "kaggle.json" ]
    then
        FLAGKAGGLE=1
        echo "Found new kaggle API key. Rewriting the previous key."
        mv ./kaggle.json ~/.kaggle/kaggle.json
        chmod 600 ~/.kaggle/kaggle.json
    fi
done

# if no kaggle API key is found, ask the user for one.
if [ $FLAGKAGGLE -eq 0 ]
then
    echo -n "Please enter your kaggle username: "
    read USERNAME
    echo -n "Please enter your kaggle API key: "
    read KAGGLEKEY
    echo "{\"username\": \"$USERNAME\", \"key\": \"$KAGGLEKEY\"}" > ./kaggle.json
fi

# start setting up the repository.
echo "Setting up the repository..."

if [ $FLAGMASTER -eq 0 ]
then
    >&2 echo "Please download the master '.zip' file of the repository from github, upload it on google colab, and then try again."
    exit 1
fi

# unzip stuff and move them to correct place.
unzip -q Pneumonia_Detection-master.zip -d .
mv Pneumonia_Detection-master/* .
rm -rf Pneumonia_Detection-master

# if data directory is present, we will ask user to either rewrite it
# or keep it as it is.
if [ $FLAGDATA -eq 1 ]
then
    echo "Data directory found. Seems like you have already downloaded the data."
    echo -n "Do you want to download and preprocess it again? [y/n]: "
    read ANS
    if [ $ANS == "y" ]
    then
        rm -rf data
        if [ $FLAGMODEL -eq 1 ]
        then
            rm -rf model
        fi
    elif [ $ANS == "n" ]
    then
        echo "Setup successful!"
        SETUP_DONE=1
    else
        >&2 echo "Command not recognized. Exiting..."
        exit 1
    fi
# delete the stale model directory, if necessary...
elif [ $FLAGMODEL -eq 1 ]
then
    echo "Deleting stale 'model' directory."
    rm -rf model
fi

if [ $SETUP_DONE -eq 0 ]
then
    # start downloading and preprocessing
    echo "Repository set up succeessfully!"
    echo "Downloading and processing the data. This may take a few minutes."

    python3 data.py

    echo "Data download and preprocessing successful!"
fi

# Let's start the training...
echo "Choose one model to train from the following list."
python3 -c "from model import pretrained_models; import pprint; pprint.pprint(list(pretrained_models.keys()))"
echo -n "Enter name of the model: "
read MODEL
echo "Choose one optimizer from the following optimizers: "
python3 -c "from main import optimizers; import pprint; pprint.pprint(list(optimizers.keys()))"
echo -n "Enter name of the optimizer: "
read OPTIM
echo -n "Enter the learning rate of the model: "
read LR
echo -n "Enter the batch size: "
read BATCH_SIZE
echo -n "Enter the number of epochs for which you wanna train your model: "
read EPOCH

python3 main.py --base_model $MODEL --optimizer $OPTIM --learning_rate $LR --batch_size $BATCH_SIZE --epoch $EPOCH --colab

echo -e "\n\n \t Training Successful! \t \n\n"
