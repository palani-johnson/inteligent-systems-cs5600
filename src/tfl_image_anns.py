########################################################
# module: tfl_image_anns.py
# authors: vladimir kulyukin
# descrption: starter code for image ANN for project 1
# bugs to vladimir kulyukin in canvas
# to install tflearn to go http://tflearn.org/installation/
########################################################

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression

## we need this to load the pickled data into Python.
def load(file_name):
    with open(file_name, "rb") as fp:
        obj = pickle.load(fp)
    return obj


PATH = "/home/palani/Projects/School/inteligent-systems-cs5600/data/project01/"
BEE1_gray_base_path = PATH + "BEE1_gray/"
BEE2_1S_gray_base_path = PATH + "BEE2_1S_gray/"
BEE4_gray_base_path = PATH + "BEE4_gray/"
NET_PATH = "data/nets_project01/image_ann/image_ann.tfl"

## let's load BEE1_gray
base_path = BEE1_gray_base_path
print("loading datasets from {}...".format(base_path))
BEE1_gray_train_X = load(base_path + "train_X.pck")
BEE1_gray_train_Y = load(base_path + "train_Y.pck")
BEE1_gray_test_X = load(base_path + "test_X.pck")
BEE1_gray_test_Y = load(base_path + "test_Y.pck")
BEE1_gray_valid_X = load(base_path + "valid_X.pck")
BEE1_gray_valid_Y = load(base_path + "valid_Y.pck")
print(BEE1_gray_train_X.shape)
print(BEE1_gray_train_Y.shape)
print(BEE1_gray_test_X.shape)
print(BEE1_gray_test_Y.shape)
print(BEE1_gray_valid_X.shape)
print(BEE1_gray_valid_Y.shape)
print("datasets from {} loaded...".format(base_path))
BEE1_gray_train_X = BEE1_gray_train_X.reshape([-1, 64, 64, 1])
BEE1_gray_test_X = BEE1_gray_test_X.reshape([-1, 64, 64, 1])

## to make sure that the dimensions of the
## examples and targets are the same.
assert BEE1_gray_train_X.shape[0] == BEE1_gray_train_Y.shape[0]
assert BEE1_gray_test_X.shape[0] == BEE1_gray_test_Y.shape[0]
assert BEE1_gray_valid_X.shape[0] == BEE1_gray_valid_Y.shape[0]

## let's load BEE2_1S_gray
base_path = BEE2_1S_gray_base_path
print("loading datasets from {}...".format(base_path))
BEE2_1S_gray_train_X = load(base_path + "train_X.pck")
BEE2_1S_gray_train_Y = load(base_path + "train_Y.pck")
BEE2_1S_gray_test_X = load(base_path + "test_X.pck")
BEE2_1S_gray_test_Y = load(base_path + "test_Y.pck")
BEE2_1S_gray_valid_X = load(base_path + "valid_X.pck")
BEE2_1S_gray_valid_Y = load(base_path + "valid_Y.pck")
print(BEE2_1S_gray_train_X.shape)
print(BEE2_1S_gray_train_Y.shape)
print(BEE2_1S_gray_test_X.shape)
print(BEE2_1S_gray_test_Y.shape)
print(BEE2_1S_gray_valid_X.shape)
print(BEE2_1S_gray_valid_Y.shape)
print("datasets from {} loaded...".format(base_path))
BEE2_1S_gray_train_X = BEE2_1S_gray_train_X.reshape([-1, 64, 64, 1])
BEE2_1S_gray_test_X = BEE2_1S_gray_test_X.reshape([-1, 64, 64, 1])

assert BEE2_1S_gray_train_X.shape[0] == BEE2_1S_gray_train_Y.shape[0]
assert BEE2_1S_gray_test_X.shape[0] == BEE2_1S_gray_test_Y.shape[0]
assert BEE2_1S_gray_valid_X.shape[0] == BEE2_1S_gray_valid_Y.shape[0]

## let's load BEE4_gray
base_path = BEE4_gray_base_path
print("loading datasets from {}...".format(base_path))
BEE4_gray_train_X = load(base_path + "train_X.pck")
BEE4_gray_train_Y = load(base_path + "train_Y.pck")
BEE4_gray_test_X = load(base_path + "test_X.pck")
BEE4_gray_test_Y = load(base_path + "test_Y.pck")
BEE4_gray_valid_X = load(base_path + "valid_X.pck")
BEE4_gray_valid_Y = load(base_path + "valid_Y.pck")
print(BEE4_gray_train_X.shape)
print(BEE4_gray_train_Y.shape)
print(BEE4_gray_test_X.shape)
print(BEE4_gray_test_Y.shape)
print(BEE4_gray_valid_X.shape)
print(BEE4_gray_valid_Y.shape)
print("datasets from {} loaded...".format(base_path))
BEE4_gray_train_X = BEE4_gray_train_X.reshape([-1, 64, 64, 1])
BEE4_gray_test_X = BEE4_gray_test_X.reshape([-1, 64, 64, 1])

assert BEE4_gray_train_X.shape[0] == BEE4_gray_train_Y.shape[0]
assert BEE4_gray_test_X.shape[0] == BEE4_gray_test_Y.shape[0]
assert BEE4_gray_valid_X.shape[0] == BEE4_gray_valid_Y.shape[0]


def __ann_template():
    net = input_data(shape=[None, 64, 64, 1])
    net = fully_connected(net, 256, activation="relu")
    net = fully_connected(net, 64, activation="relu")
    net = fully_connected(net, 64, activation="relu")
    net = dropout(net, 0.7, name="dropout")
    net = fully_connected(net, 2, activation="softmax")
    net = regression(
        net,
        optimizer="sgd",
        loss="categorical_crossentropy",
        learning_rate=0.01,
    )
    return tflearn.DNN(net)


def make_image_ann_model():
    return __ann_template()


def load_image_ann_model(model_path):
    tf.compat.v1.reset_default_graph()
    model = __ann_template()
    model.load(model_path)
    return model


### test a tfl network model on valid_X and valid_Y.
def test_tfl_image_ann_model(network_model, valid_X, valid_Y):
    results = []
    for i in range(len(valid_X)):
        prediction = network_model.predict(valid_X[i].reshape([-1, 64, 64, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == np.argmax(valid_Y[i]))
    return float(sum((np.array(results) == True))) / float(len(results))


###  train a tfl model on train_X, train_Y, test_X, test_Y.
def train_tfl_image_ann_model(
    model, train_X, train_Y, test_X, test_Y, num_epochs=2, batch_size=10
):
    tf.compat.v1.reset_default_graph()
    model.fit(
        train_X,
        train_Y,
        n_epoch=num_epochs,
        shuffle=True,
        validation_set=(test_X, test_Y),
        show_metric=True,
        batch_size=batch_size,
        run_id="image_ann_model",
    )


### validating is testing on valid_X and valid_Y.
def validate_tfl_image_ann_model(model, valid_X, valid_Y):
    return test_tfl_image_ann_model(model, valid_X, valid_Y)


if __name__ == "__main__":
    img_ann = (
        make_image_ann_model()
        if input("New?[y/n] ").upper() == "Y"
        else load_image_ann_model(NET_PATH)
    )

    train_tfl_image_ann_model(
        img_ann,
        np.concatenate(
            (BEE1_gray_train_X, BEE2_1S_gray_train_X, BEE4_gray_train_X),
            axis=0,
        ),
        np.concatenate(
            (BEE1_gray_train_Y, BEE2_1S_gray_train_Y, BEE4_gray_train_Y),
            axis=0,
        ),
        np.concatenate(
            (BEE1_gray_test_X, BEE2_1S_gray_test_X, BEE4_gray_test_X),
            axis=0,
        ),
        np.concatenate(
            (BEE1_gray_test_Y, BEE2_1S_gray_test_Y, BEE4_gray_test_Y),
            axis=0,
        ),
        num_epochs=50,
        batch_size=25,
    )

    print(
        "BEE1_gray acc: {}".format(
            validate_tfl_image_ann_model(
                img_ann,
                BEE1_gray_valid_X,
                BEE1_gray_valid_Y,
            )
        )
    )

    print(
        "BEE2_1S_gray acc: {}".format(
            validate_tfl_image_ann_model(
                img_ann,
                BEE2_1S_gray_valid_X,
                BEE2_1S_gray_valid_Y,
            )
        )
    )

    print(
        "BEE4_gray acc: {}".format(
            validate_tfl_image_ann_model(
                img_ann,
                BEE4_gray_valid_X,
                BEE4_gray_valid_Y,
            )
        )
    )

    if input("Save?[y/n] ").upper() == "Y":
        img_ann.save(NET_PATH)
