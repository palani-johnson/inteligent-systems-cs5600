# Project 2

This repository contains a copy of cell instance segmentation model that I trained for the [Sartorius - Cell Instance Segmentation competition](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/data).

## Setup
I use [Poetry](https://python-poetry.org/docs/) to help manage the version dependencies. If you dont have poetry installed you can follow the instructions [found here](https://python-poetry.org/docs/#installation). Once installed you can run the following to install all the required dependencies. 

```
poetry install 
```

If you want poetry to create a virtual environment in this repo you should run this command BEFORE you run the above.
```
poetry config virtualenvs.in-project true
```

## Starting Jupyter
If you have Jupyter already set up you can simply use your perfered editor. I like VS Code with the Microsoft Jupyter notebook extension. Make sure you select the correct kernal. 

## Downloading the dataset
You can use the [Kaggle API](https://www.kaggle.com/docs/api) to download the dataset. Run the following once you have everything set up:
```
mkdir sartorius-cell-instance-segmentation && cd sartorius-cell-instance-segmentation && \
kaggle competitions download -c sartorius-cell-instance-segmentation && cd ..
```

Next extract the contents of the zip file into that folder.

## Downloading the model
The model is titled `unet_keras_model.h5` and can be found [here](https://www.kaggle.com/palanijohnson/sarcompinputmodel). Move the file into the root of this directory.

## Make the output folder
```
mkdir output
```

## Running the notebook
The notebook can now be run. It is currently set to load in the existing model and not perform any extra training on it. Running the model like this will show you some example masks created by it. If you want to see how it trains you can change the constant set under 4. constants in the notebook called `TRAIN`. Lower the epochs accordingly as it is currently set to 30. 


