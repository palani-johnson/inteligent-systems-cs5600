############################################################
## module: load_data.py
## description: this is an aux file for keras_img_nets.py
## authors: vladimir Kulyukin
#############################################################

import glob
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
from skimage import io, color, transform
from pickle import dump, load

basepath = "/home/palani/Projects/School/inteligent-systems-cs5600/data/project01/"
IMG_WIDTH = 64
IMG_HEIGHT = 64

### We need change these global variables as necessary.
### For example, when we work with BEE1, we need to
### change dataset to 'BEE1', num_classes=2.
### For BEE4, dataset='BEE4' num_classes=2.

dataset = "BEE3"
shadow = True
num_classes = 3

BEE = 0  ## if image contains a bee, it's categorical label is 0;
NO_BEE = 1  ## if image doesn't contain a bee, it's categorical label is 1;
SHADOW_BEE = 2  ## if image contains a bee, it's categorical label is 2.

### shadown=False this is always False
def get_file_paths(dataset, divsion, shadow=False):
    """
    get_file_paths('BEE1', 'training', shadow=False)
    division can be 'training', 'testing', or 'validation'.
    """
    global basepath, BEE, NO_BEE, SHADOW_BEE
    file_paths = []
    bees = glob.glob(basepath + dataset + "/" + divsion + "/bee/*/*.png")
    print(basepath + dataset + "/" + divsion + "/bee/*/*.png")
    nonbees = glob.glob(basepath + dataset + "/" + divsion + "/no_bee/*/*.png")
    ### BEE is classified as 0.
    for file in bees:
        file_paths.append((file, BEE))
    ### NO_BEE is classifed as 1.
    for file in nonbees:
        file_paths.append((file, NO_BEE))
    ### SHADOW_BEE is classified as 2.
    if shadow:
        shadowbees = glob.glob(
            basepath + dataset + "/" + divsion + "/shadow_bee/*/*.png"
        )
        for file in shadowbees:
            file_paths.append((file, SHADOW_BEE))
    file_paths = shuffle(file_paths)
    return file_paths


def image_process(
    file_paths,
    gray_scale=False,
    normalize=True,
    to_categorical=True,
    flatten=False,
    save_name=None,
):
    """
    image_process() is used to convert images for input to keras nets.
    file_paths is an array of string file paths to individual images.
    to_categorial, when True, is being used for tensorfolow one-hot encoding (e.g., [0, 0, 1] if there
    are 3 classes.) This function is OpenCV independent; it is dependent on
    skimage.io, skimage.color, and skiimage.transform.
    """
    global IMG_HEIGHT, IMG_WIDTH, num_classes
    batchX = []
    batchY = []

    if save_name:
        try:
            print(f"***** Loading {save_name}_X.pk *****")
            with open(f"{basepath}/{dataset}/{save_name}_X.pk", "rb") as f:
                batchX = load(f)
            print(f"***** Loading {save_name}_Y.pk *****")
            with open(f"{basepath}/{dataset}/{save_name}_Y.pk", "rb") as f:
                batchY = load(f)
            return batchX, batchY
        except:
            print("load failed")
    print("**************** CHECK 0000 ****************")
    COUNT = 0
    for i in range(len(file_paths)):
        if i % 5000 == 0:
            ### Debugging; comment out if needed
            print("loaded {}...".format(i))
        ### 1. Read the image from a file path
        image = io.imread(file_paths[i][0])
        ### 2. Grayscale if necessary
        if gray_scale:
            image = color.rgb2gray(image)
        ### 3. Normalize the pixels if necessary
        if normalize:
            image = image / 255.0
        ### 4. Resize as required
        image = transform.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        ### 5. Flatten
        image = np.array(image[:, :, :3])
        ### 6. put the image array into batchX
        batchX.append(image)
        ### 7. put the categorical label into batchY
        batchY.append(file_paths[i][1])
    if normalize:
        batchX = np.array(batchX, dtype="float32")
    else:
        batchX = np.array(batchX, dtype="int64")
    batchY = np.array(batchY)
    if to_categorical:
        batchY = tf.keras.utils.to_categorical(batchY, num_classes=num_classes)
    if flatten:
        batchX = batchX.reshape(-1, IMG_HEIGHT * IMG_WIDTH)
    if save_name:
        with open(f"{basepath}/{dataset}/{save_name}_X.pk", "wb") as f:
            dump(batchX, f)
        with open(f"{basepath}/{dataset}/{save_name}_Y.pk", "wb") as f:
            dump(batchY, f)
    return batchX, batchY


### train_file_paths are an array of 2-tuples (path, category).
train_file_paths = get_file_paths(dataset, "training", shadow=shadow)
train_X, train_Y = image_process(train_file_paths, save_name="train")
print(train_X.shape[0])
print(train_Y.shape[0])
assert train_X.shape[0] == train_Y.shape[0]

test_file_paths = get_file_paths(dataset, "testing", shadow=shadow)
test_X, test_Y = image_process(test_file_paths, save_name="test")
assert test_X.shape[0] == test_Y.shape[0]

valid_file_paths = get_file_paths(dataset, "validation", shadow=shadow)
valid_X, valid_Y = image_process(valid_file_paths, save_name="valid")
assert valid_X.shape[0] == valid_Y.shape[0]
