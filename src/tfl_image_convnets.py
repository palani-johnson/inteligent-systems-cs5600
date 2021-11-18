########################################################
# module: tfl_image_convnets.py
# authors: vladimir kulyukin
# descrption: starter code for image ConvNets for Project 1
# to install tflearn to go http://tflearn.org/installation/
########################################################

import pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import dropout, input_data, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

## we need this to load the pickled data into Python.
def load(file_name):
    with open(file_name, "rb") as fp:
        obj = pickle.load(fp)
    return obj


PATH = "/home/palani/Projects/School/inteligent-systems-cs5600/data/project01/"
BEE1_path = PATH + "BEE1/"
BEE2_1S_path = PATH + "BEE2_1S/"
BEE4_path = PATH + "BEE4/"
CONVNET_PATH = "data/nets_project01/image_cn/image_cn.tfl"

## let's load BEE1
base_path = BEE1_path
print("loading datasets from {}...".format(base_path))
BEE1_train_X = load(base_path + "train_X.pck")
BEE1_train_Y = load(base_path + "train_Y.pck")
BEE1_test_X = load(base_path + "test_X.pck")
BEE1_test_Y = load(base_path + "test_Y.pck")
BEE1_valid_X = load(base_path + "valid_X.pck")
BEE1_valid_Y = load(base_path + "valid_Y.pck")
print(BEE1_train_X.shape)
print(BEE1_train_Y.shape)
print(BEE1_test_X.shape)
print(BEE1_test_Y.shape)
print(BEE1_valid_X.shape)
print(BEE1_valid_Y.shape)
print("datasets from {} loaded...".format(base_path))
BEE1_train_X = BEE1_train_X.reshape([-1, 64, 64, 3])
BEE1_test_X = BEE1_test_X.reshape([-1, 64, 64, 3])

## to make sure that the dimensions of the
## examples and targets are the same.
assert BEE1_train_X.shape[0] == BEE1_train_Y.shape[0]
assert BEE1_test_X.shape[0] == BEE1_test_Y.shape[0]
assert BEE1_valid_X.shape[0] == BEE1_valid_Y.shape[0]

## let's load BEE2_1S
base_path = BEE2_1S_path
print("loading datasets from {}...".format(base_path))
BEE2_1S_train_X = load(base_path + "train_X.pck")
BEE2_1S_train_Y = load(base_path + "train_Y.pck")
BEE2_1S_test_X = load(base_path + "test_X.pck")
BEE2_1S_test_Y = load(base_path + "test_Y.pck")
BEE2_1S_valid_X = load(base_path + "valid_X.pck")
BEE2_1S_valid_Y = load(base_path + "valid_Y.pck")
print(BEE2_1S_train_X.shape)
print(BEE2_1S_train_Y.shape)
print(BEE2_1S_test_X.shape)
print(BEE2_1S_test_Y.shape)
print(BEE2_1S_valid_X.shape)
print(BEE2_1S_valid_Y.shape)
print("datasets from {} loaded...".format(base_path))
BEE2_1S_train_X = BEE2_1S_train_X.reshape([-1, 64, 64, 3])
BEE2_1S_test_X = BEE2_1S_test_X.reshape([-1, 64, 64, 3])

assert BEE2_1S_train_X.shape[0] == BEE2_1S_train_Y.shape[0]
assert BEE2_1S_test_X.shape[0] == BEE2_1S_test_Y.shape[0]
assert BEE2_1S_valid_X.shape[0] == BEE2_1S_valid_Y.shape[0]

## let's load BEE4
base_path = BEE4_path
print("loading datasets from {}...".format(base_path))
BEE4_train_X = load(base_path + "train_X.pck")
BEE4_train_Y = load(base_path + "train_Y.pck")
BEE4_test_X = load(base_path + "test_X.pck")
BEE4_test_Y = load(base_path + "test_Y.pck")
BEE4_valid_X = load(base_path + "valid_X.pck")
BEE4_valid_Y = load(base_path + "valid_Y.pck")
print(BEE4_train_X.shape)
print(BEE4_train_Y.shape)
print(BEE4_test_X.shape)
print(BEE4_test_Y.shape)
print(BEE4_valid_X.shape)
print(BEE4_valid_Y.shape)
print("datasets from {} loaded...".format(base_path))
BEE4_train_X = BEE4_train_X.reshape([-1, 64, 64, 3])
BEE4_test_X = BEE4_test_X.reshape([-1, 64, 64, 3])

assert BEE4_train_X.shape[0] == BEE4_train_Y.shape[0]
assert BEE4_test_X.shape[0] == BEE4_test_Y.shape[0]
assert BEE4_valid_X.shape[0] == BEE4_valid_Y.shape[0]


def __conv_template():
    net = input_data(shape=[None, 64, 64, 3])
    net = conv_2d(net, nb_filter=16, filter_size=16, activation="relu")
    net = max_pool_2d(net, 4)
    net = conv_2d(net, nb_filter=4, filter_size=4, activation="relu")
    net = max_pool_2d(net, 4)
    net = fully_connected(net, 64, activation="relu")
    net = dropout(net, 0.7)
    net = fully_connected(net, 2, activation="softmax")
    net = regression(
        net,
        optimizer="sgd",
        loss="categorical_crossentropy",
        learning_rate=0.05,
    )
    return tflearn.DNN(net)


def make_image_convnet_model():
    return __conv_template()


def load_image_convnet_model(model_path):
    tf.compat.v1.reset_default_graph()
    model = __conv_template()
    model.load(model_path)
    return model


def test_tfl_image_convnet_model(network_model, validX, validY):
    results = []
    for i in range(len(validX)):
        prediction = network_model.predict(validX[i].reshape([-1, 64, 64, 3]))
        results.append(np.argmax(prediction, axis=1)[0] == np.argmax(validY[i]))
    return float(sum((np.array(results) == True))) / float(len(results))


###  train a tfl convnet model on train_X, train_Y, test_X, test_Y.
def train_tfl_image_convnet_model(
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
        run_id="image_cn_model",
    )


### validating is testing on valid_X and valid_Y.
def validate_tfl_image_convnet_model(model, valid_X, valid_Y):
    return test_tfl_image_convnet_model(model, valid_X, valid_Y)


if __name__ == "__main__":
    img_cn = (
        make_image_convnet_model()
        if input("New?[y/n] ").upper() == "Y"
        else load_image_convnet_model(CONVNET_PATH)
    )

    train_tfl_image_convnet_model(
        img_cn,
        np.concatenate(
            (BEE1_train_X, BEE2_1S_train_X, BEE4_train_X),
            axis=0,
        ),
        np.concatenate(
            (BEE1_train_Y, BEE2_1S_train_Y, BEE4_train_Y),
            axis=0,
        ),
        np.concatenate(
            (BEE1_test_X, BEE2_1S_test_X, BEE4_test_X),
            axis=0,
        ),
        np.concatenate(
            (BEE1_test_Y, BEE2_1S_test_Y, BEE4_test_Y),
            axis=0,
        ),
        num_epochs=10,
        batch_size=50,
    )

    print(
        "BEE1 acc: {}".format(
            validate_tfl_image_convnet_model(
                img_cn,
                BEE1_valid_X,
                BEE1_valid_Y,
            )
        )
    )

    print(
        "BEE2_1S acc: {}".format(
            validate_tfl_image_convnet_model(
                img_cn,
                BEE2_1S_valid_X,
                BEE2_1S_valid_Y,
            )
        )
    )

    print(
        "BEE4 acc: {}".format(
            validate_tfl_image_convnet_model(
                img_cn,
                BEE4_valid_X,
                BEE4_valid_Y,
            )
        )
    )

    if input("Save?[y/n] ").upper() == "Y":
        img_cn.save(CONVNET_PATH)
