########################################################
# module: tfl_audio_anns.py
# authors: vladimir kulyukin
# descrption: starter code for audio ANN for project 1
# to install tflearn to go http://tflearn.org/installation/
########################################################

import pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import dropout, input_data, fully_connected
from tflearn.layers.estimator import regression

## we need this to load the pickled data into Python.
def load(file_name):
    with open(file_name, "rb") as fp:
        obj = pickle.load(fp)
    return obj


PATH = "/home/palani/Projects/School/inteligent-systems-cs5600/data/project01/"
BUZZ1_base_path = PATH + "BUZZ1/"
BUZZ2_base_path = PATH + "BUZZ2/"
BUZZ3_base_path = PATH + "BUZZ3/"
NET_PATH = "data/nets_project01/aud_ann/aud_ann.tfl"

## let's load BUZZ1
base_path = BUZZ1_base_path
print("loading datasets from {}...".format(base_path))
BUZZ1_train_X = load(base_path + "train_X.pck")
BUZZ1_train_Y = load(base_path + "train_Y.pck")
BUZZ1_test_X = load(base_path + "test_X.pck")
BUZZ1_test_Y = load(base_path + "test_Y.pck")
BUZZ1_valid_X = load(base_path + "valid_X.pck")
BUZZ1_valid_Y = load(base_path + "valid_Y.pck")
print(BUZZ1_train_X.shape)
print(BUZZ1_train_Y.shape)
print(BUZZ1_test_X.shape)
print(BUZZ1_test_Y.shape)
print(BUZZ1_valid_X.shape)
print(BUZZ1_valid_Y.shape)
print("datasets from {} loaded...".format(base_path))
BUZZ1_train_X = BUZZ1_train_X.reshape([-1, 4000, 1, 1])
BUZZ1_test_X = BUZZ1_test_X.reshape([-1, 4000, 1, 1])

## to make sure that the dimensions of the
## examples and targets are the same.
assert BUZZ1_train_X.shape[0] == BUZZ1_train_Y.shape[0]
assert BUZZ1_test_X.shape[0] == BUZZ1_test_Y.shape[0]
assert BUZZ1_valid_X.shape[0] == BUZZ1_valid_Y.shape[0]

## let's load BUZZ2
base_path = BUZZ2_base_path
print("loading datasets from {}...".format(base_path))
BUZZ2_train_X = load(base_path + "train_X.pck")
BUZZ2_train_Y = load(base_path + "train_Y.pck")
BUZZ2_test_X = load(base_path + "test_X.pck")
BUZZ2_test_Y = load(base_path + "test_Y.pck")
BUZZ2_valid_X = load(base_path + "valid_X.pck")
BUZZ2_valid_Y = load(base_path + "valid_Y.pck")
print(BUZZ2_train_X.shape)
print(BUZZ2_train_Y.shape)
print(BUZZ2_test_X.shape)
print(BUZZ2_test_Y.shape)
print(BUZZ2_valid_X.shape)
print(BUZZ2_valid_Y.shape)
print("datasets from {} loaded...".format(base_path))
BUZZ2_train_X = BUZZ2_train_X.reshape([-1, 4000, 1, 1])
BUZZ2_test_X = BUZZ2_test_X.reshape([-1, 4000, 1, 1])

assert BUZZ2_train_X.shape[0] == BUZZ2_train_Y.shape[0]
assert BUZZ2_test_X.shape[0] == BUZZ2_test_Y.shape[0]
assert BUZZ2_valid_X.shape[0] == BUZZ2_valid_Y.shape[0]

## let's load BUZZ3
base_path = BUZZ3_base_path
print("loading datasets from {}...".format(base_path))
BUZZ3_train_X = load(base_path + "train_X.pck")
BUZZ3_train_Y = load(base_path + "train_Y.pck")
BUZZ3_test_X = load(base_path + "test_X.pck")
BUZZ3_test_Y = load(base_path + "test_Y.pck")
BUZZ3_valid_X = load(base_path + "valid_X.pck")
BUZZ3_valid_Y = load(base_path + "valid_Y.pck")
print(BUZZ3_train_X.shape)
print(BUZZ3_train_Y.shape)
print(BUZZ3_test_X.shape)
print(BUZZ3_test_Y.shape)
print(BUZZ3_valid_X.shape)
print(BUZZ3_valid_Y.shape)
print("datasets from {} loaded...".format(base_path))
BUZZ3_train_X = BUZZ3_train_X.reshape([-1, 4000, 1, 1])
BUZZ3_test_X = BUZZ3_test_X.reshape([-1, 4000, 1, 1])

assert BUZZ3_train_X.shape[0] == BUZZ3_train_Y.shape[0]
assert BUZZ3_test_X.shape[0] == BUZZ3_test_Y.shape[0]
assert BUZZ3_valid_X.shape[0] == BUZZ3_valid_Y.shape[0]


def __ann_template():
    net = input_data(shape=[None, 4000, 1, 1])
    net = fully_connected(net, 32, activation="relu")
    net = fully_connected(net, 32, activation="relu")
    net = dropout(net, 0.7, name="dropout")
    net = fully_connected(net, 3, activation="softmax")
    net = regression(
        net,
        optimizer="sgd",
        loss="categorical_crossentropy",
        learning_rate=0.025,
    )
    return tflearn.DNN(net)


def make_audio_ann_model():
    return __ann_template()


def load_audio_ann_model(model_path):
    model = __ann_template()
    model.load(model_path)
    return model


### test a tfl network model on valid_X and valid_Y.
def test_tfl_audio_ann_model(network_model, valid_X, valid_Y):
    results = []
    for i in range(len(valid_X)):
        prediction = network_model.predict(valid_X[i].reshape([-1, 4000, 1, 1]))
        results.append(np.argmax(prediction, axis=1)[0] == np.argmax(valid_Y[i]))
    return float(sum((np.array(results) == True))) / float(len(results))


###  train a tfl model on train_X, train_Y, test_X, test_Y.
def train_tfl_audio_ann_model(
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
        run_id="audio_ann_model",
    )


### validating is testing on valid_X and valid_Y.
def validate_tfl_audio_ann_model(model, valid_X, valid_Y):
    return test_tfl_audio_ann_model(model, valid_X, valid_Y)


if __name__ == "__main__":
    aud_ann = (
        make_audio_ann_model()
        if input("New?[y/n] ").upper() == "Y"
        else load_audio_ann_model(NET_PATH)
    )

    train_tfl_audio_ann_model(
        aud_ann,
        np.concatenate(
            (BUZZ1_train_X, BUZZ2_train_X, BUZZ3_train_X),
            axis=0,
        ),
        np.concatenate(
            (BUZZ1_train_Y, BUZZ2_train_Y, BUZZ3_train_Y),
            axis=0,
        ),
        np.concatenate(
            (BUZZ1_test_X, BUZZ2_test_X, BUZZ3_test_X),
            axis=0,
        ),
        np.concatenate(
            (BUZZ1_test_Y, BUZZ2_test_Y, BUZZ3_test_Y),
            axis=0,
        ),
        num_epochs=50,
        batch_size=20,
    )

    print(
        "BUZZ1 acc: {}".format(
            validate_tfl_audio_ann_model(
                aud_ann,
                BUZZ1_valid_X,
                BUZZ1_valid_Y,
            )
        )
    )

    print(
        "BUZZ2 acc: {}".format(
            validate_tfl_audio_ann_model(
                aud_ann,
                BUZZ2_valid_X,
                BUZZ2_valid_Y,
            )
        )
    )

    print(
        "BUZZ3 acc: {}".format(
            validate_tfl_audio_ann_model(
                aud_ann,
                BUZZ3_valid_X,
                BUZZ3_valid_Y,
            )
        )
    )

    if input("Save?[y/n] ").upper() == "Y":
        aud_ann.save(NET_PATH)
