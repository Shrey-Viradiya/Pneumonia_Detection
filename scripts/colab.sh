cd /content/Pneumonia_Detection

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
FLAGDATA=0
FLAGMODEL=0
SETUP_DONE=0

for FILENAME in $CURRDIR
do
    # check if data directory is present
    if [ $FILENAME == "data" ]
    then
        FLAGDATA=1
    # check if model is present
    elif [ $FILENAME == "model_objects" ]
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
    read KAGGLEUSERNAME
    echo -n "Please enter your kaggle API key: "
    read KAGGLEKEY
    echo "{\"username\": \"$KAGGLEUSERNAME\", \"key\": \"$KAGGLEKEY\"}" > ./kaggle.json
fi

# start setting up the repository.
echo "Setting up the repository..."

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
    elif [ $ANS == "n" ]
    then
        SETUP_DONE=1
    else
        >&2 echo "Command not recognized. Exiting..."
        exit 1
    fi
fi

echo "Data Setup Done!"

if [ $FLAGMODEL -eq 1 ]
then
    echo -n "A directory of saved models found. Do you want to use it to load models? [y/n] : "
    read ANS
    if [ $ANS == "n" ]
    then
        rm -rf model_objects
        rm -rf model_metadata
    elif [ $ANS == "y" ]
    then
        echo "Ok, I will use the models present in model_objects directory!"
    else
        >&2 echo "Command not recognized. Exiting..."
        exit 1
    fi
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

python3 main.py --base_model $MODEL --optimizer $OPTIM --learning_rate $LR --batch_size $BATCH_SIZE --epoch $EPOCH

echo -e "\n\n \t Training Successful! \t \n\n"
