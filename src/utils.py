# /usr/bin/python

###########################################
# Unit Tests for Assignment 5
# bugs to vladimir kulyukin in canvas
###########################################

import unittest
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist
from hw05 import *

### Let's load MNIST and reshape train, test, and validation sets.
X, Y, testX, testY = mnist.load_data(one_hot=True)
testX, testY = tflearn.data_utils.shuffle(testX, testY)
trainX, trainY = X[0:50000], Y[0:50000]
validX, validY = X[50000:], Y[50000:]
validX, validY = tflearn.data_utils.shuffle(validX, validY)
trainX = trainX.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
validX = validX.reshape([-1, 28, 28, 1])


NET_PATH = "data/nets_hw05/"


class cs5600_6600_f21_hw05_uts(unittest.TestCase):
    # def test_ut01(self):
    #     tf.compat.v1.reset_default_graph()
    #     model = make_tfl_mnist_convnet()
    #     model_name = "my_tfl_mnist_convnet"
    #     assert model is not None
    #     fit_tfl_model(
    #         model, trainX, trainY, testX, testY, model_name, NET_PATH, n_epoch=5, mbs=10
    #     )

    # def test_ut02(self):
    #     tf.compat.v1.reset_default_graph()
    #     model = load_tfl_mnist_convnet(NET_PATH + "my_tfl_mnist_convnet")
    #     assert model is not None
    #     i = np.random.randint(0, len(validX) - 1)
    #     prediction = model.predict(validX[i].reshape([-1, 28, 28, 1]))
    #     print("raw prediction   = {}".format(prediction))
    #     print("raw ground truth = {}".format(validY[i]))
    #     prediction = np.argmax(prediction, axis=1)[0]
    #     ground_truth = np.argmax(validY[i])
    #     print("ground truth = {}".format(ground_truth))
    #     print("prediction   = {}".format(prediction))
    #     print(prediction == ground_truth)

    # def test_ut03(self):
    #     tf.compat.v1.reset_default_graph()
    #     model = load_tfl_mnist_convnet(NET_PATH + "my_tfl_mnist_convnet")
    #     assert model is not None
    #     acc = test_tfl_model(model, validX, validY)
    #     print("tfl mnist model acc = {}".format(acc))

    # def test_ut04(self):
    #     tf.compat.v1.reset_default_graph()
    #     deeper_model = make_deeper_tfl_mnist_convnet()
    #     model_name = "my_deeper_tfl_mnist_model"
    #     assert deeper_model is not None
    #     fit_tfl_model(
    #         deeper_model,
    #         trainX,
    #         trainY,
    #         testX,
    #         testY,
    #         model_name,
    #         NET_PATH,
    #         n_epoch=50,
    #         mbs=10,
    #     )

    def test_ut05(self):
        tf.compat.v1.reset_default_graph()
        model_name = "my_deeper_tfl_mnist_model"
        deeper_model = load_deeper_tfl_mnist_convnet(NET_PATH + model_name)
        assert deeper_model is not None
        acc = test_tfl_model(deeper_model, validX, validY)
        print("tfl mnist deeper convnet acc = {}".format(acc))

    # def test_ut06(self):
    #     tf.compat.v1.reset_default_graph()
    #     shallow_model = make_shallow_tfl_mnist_ann()
    #     model_name = "my_shallow_tfl_mnist_ann"
    #     assert shallow_model is not None
    #     fit_tfl_model(
    #         shallow_model,
    #         trainX,
    #         trainY,
    #         testX,
    #         testY,
    #         model_name,
    #         NET_PATH,
    #         n_epoch=50,
    #         mbs=10,
    #     )

    def test_ut07(self):
        tf.compat.v1.reset_default_graph()
        model_name = "my_shallow_tfl_mnist_ann"
        shallow_model = load_shallow_tfl_mnist_ann(NET_PATH + model_name)
        assert shallow_model is not None
        acc = test_tfl_model(shallow_model, validX, validY)
        print("shallow tfl mnist ann acc = {}".format(acc))


if __name__ == "__main__":
    unittest.main()
