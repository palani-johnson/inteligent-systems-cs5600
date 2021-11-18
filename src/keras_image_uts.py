########################################################
# module: keras_image_uts.py
# authors: vladimir kulyukin
# descrption: Unit tests for Project 1 ConvNet
# trained with Keras.
########################################################

import tensorflow as tf
from keras_image_convnets import *
import unittest


class keras_image_uts(unittest.TestCase):
    def test_keras_convnet_bee4(self):
        km = load_keras_model()
        valid_loss, valid_acc = km.evaluate(valid_X, valid_Y, verbose=0)
        print("Keras BEE4 valid accuracy: {:5.2f}%".format(100 * valid_acc))
        print("Keras BEE4 valid loss: {:5.2f}%".format(100 * valid_loss))


### ================ Unit Tests ====================

if __name__ == "__main__":
    unittest.main()
