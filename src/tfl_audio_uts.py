########################################################
# module: tfl_audio_uts.py
# authors: vladimir kulyukin
# descrption: Unit tests for Project 1 audio ANN and ConvNet
# trained with TFLearn.
# bugs to vladimir kulyukin in canvas
# to install tflearn to go http://tflearn.org/installation/
########################################################

from tfl_audio_anns import *
from tfl_audio_convnets import *
import tensorflow as tf
import unittest


class tfl_audio_uts(unittest.TestCase):
    def test_tfl_audio_ann_buzz1(self):
        tf.compat.v1.reset_default_graph()
        an = load_audio_ann_model(NET_PATH)
        vacc = validate_tfl_audio_ann_model(an, BUZZ1_valid_X, BUZZ1_valid_Y)
        print("\n\n\n\n\n**** Ann valid. acc on BUZZ1 = {}\n\n\n\n\n".format(vacc))

    def test_tfl_audio_ann_buzz2(self):
        tf.compat.v1.reset_default_graph()
        an = load_audio_ann_model(NET_PATH)
        vacc = validate_tfl_audio_ann_model(an, BUZZ2_valid_X, BUZZ2_valid_Y)
        print("\n\n\n\n\n**** Ann valid. acc on BUZZ2 = {}\n\n\n\n\n".format(vacc))

    def test_tfl_audio_ann_buzz3(self):
        tf.compat.v1.reset_default_graph()
        an = load_audio_ann_model(NET_PATH)
        vacc = validate_tfl_audio_ann_model(an, BUZZ3_valid_X, BUZZ3_valid_Y)
        print("\n\n\n\n\n**** Ann valid. acc on BUZZ3 = {}\n\n\n\n\n".format(vacc))

    def test_tfl_audio_convnet_buzz1(self):
        tf.compat.v1.reset_default_graph()
        cn = load_audio_convnet_model(CONVNET_PATH)
        vacc = validate_tfl_audio_convnet_model(cn, BUZZ1_valid_X, BUZZ1_valid_Y)
        print("\n\n\n\n\n**** CN valid. acc on BUZZ1 = {}\n\n\n\n\n".format(vacc))

    def test_tfl_audio_convnet_buzz2(self):
        tf.compat.v1.reset_default_graph()
        cn = load_audio_convnet_model(CONVNET_PATH)
        vacc = validate_tfl_audio_convnet_model(cn, BUZZ2_valid_X, BUZZ2_valid_Y)
        print("\n\n\n\n\n**** CN valid. acc on BUZZ2 = {}\n\n\n\n\n".format(vacc))

    def test_tfl_audio_convnet_buzz3(self):
        tf.compat.v1.reset_default_graph()
        cn = load_audio_convnet_model(CONVNET_PATH)
        vacc = validate_tfl_audio_convnet_model(cn, BUZZ3_valid_X, BUZZ3_valid_Y)
        print("\n\n\n\n\n**** CN valid. acc on BUZZ3 = {}\n\n\n\n\n".format(vacc))


### ================ Unit Tests ====================

if __name__ == "__main__":
    unittest.main()
