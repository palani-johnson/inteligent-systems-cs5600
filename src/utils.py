# /usr/bin/python

###########################################
# Unit Tests for Assignment 4
# bugs to vladimir kulyukin via canvas
###########################################

import unittest

from ann import *
from hw04 import *
from mnist_loader import load_data_wrapper

## change this directory accordingly.
DIR_PATH = "data/nets_hw04/"

train_d, valid_d, test_d = load_data_wrapper()


class cs5600_6600_f21_hw02_uts(unittest.TestCase):
    # def test_ut01(self):
    #     global train_d
    #     net = ann([784, 10, 10], cost=CrossEntropyCost)
    #     net_stats = net.mini_batch_sgd(
    #         train_d,
    #         5,
    #         10,
    #         0.5,
    #         lmbda=5.0,
    #         evaluation_data=valid_d,
    #         monitor_evaluation_cost=True,
    #         monitor_evaluation_accuracy=True,
    #         monitor_training_cost=True,
    #         monitor_training_accuracy=True,
    #     )
    #     plot_costs(net_stats[0], net_stats[2], 5)

    # def test_ut02(self):
    #     global train_d
    #     net = ann([784, 10, 10], cost=CrossEntropyCost)
    #     net_stats = net.mini_batch_sgd(
    #         train_d,
    #         10,
    #         10,
    #         0.5,
    #         lmbda=0.5,
    #         evaluation_data=valid_d,
    #         monitor_evaluation_cost=True,
    #         monitor_evaluation_accuracy=True,
    #         monitor_training_cost=True,
    #         monitor_training_accuracy=True,
    #     )
    #     plot_accuracies(net_stats[1], net_stats[3], 10)

    # def test_ut03(self):
    #     print("*** Running Unit Test 3 ***")
    #     d = collect_1_hidden_layer_net_stats(
    #         10, 11, CrossEntropyCost, 2, 10, 0.1, 0.0, train_d, test_d
    #     )
    #     assert len(d) == 2
    #     assert len(d[10]) == 4
    #     assert len(d[11]) == 4
    #     assert len(d[10][0]) == 2
    #     assert len(d[10][1]) == 2
    #     assert len(d[10][2]) == 2
    #     assert len(d[10][3]) == 2
    #     assert len(d[11][0]) == 2
    #     assert len(d[11][1]) == 2
    #     assert len(d[11][2]) == 2
    #     assert len(d[11][3]) == 2
    #     for k, v in d.items():
    #         print("{} --> {}".format(k, v))

    # def test_ut04(self):
    #     print("*** Running Unit Test 4 ***")
    #     d = collect_2_hidden_layer_net_stats(
    #         2, 3, CrossEntropyCost, 2, 10, 0.1, 0.0, train_d, test_d
    #     )
    #     assert len(d) == 4
    #     assert len(d["2_2"]) == 4
    #     assert len(d["2_3"]) == 4
    #     assert len(d["3_2"]) == 4
    #     assert len(d["3_3"]) == 4
    #     for k, v in d.items():
    #         print("{} --> {}".format(k, v))

    # def test_ut05(self):
    #     print("*** Running Unit Test 5 ***")
    #     d = collect_3_hidden_layer_net_stats(
    #         2, 3, CrossEntropyCost, 2, 10, 0.1, 0.0, train_d, test_d
    #     )
    #     assert len(d) == 8
    #     assert len(d["2_2_2"]) == 4
    #     assert len(d["2_3_2"]) == 4
    #     assert len(d["3_2_2"]) == 4
    #     assert len(d["3_3_3"]) == 4
    #     for k, v in d.items():
    #         print("{} --> {}".format(k, v))
    #     return d

    def test_ut06(self, evaluation_data=test_d):
        net1 = load(DIR_PATH + "net1.json")
        net2 = load(DIR_PATH + "net2.json")
        net3 = load(DIR_PATH + "net3.json")
        print(
            "net1's accuracy on evaluation data: {} / {}".format(
                net1.accuracy(evaluation_data), len(evaluation_data)
            )
        )
        print(
            "net2's accuracy on evaluation data: {} / {}".format(
                net2.accuracy(evaluation_data), len(evaluation_data)
            )
        )
        print(
            "net3's accuracy on evaluation data: {} / {}".format(
                net3.accuracy(evaluation_data), len(evaluation_data)
            )
        )


if __name__ == "__main__":
    unittest.main()
