#!/usr/bin/python

#########################################
# module: cs5600_6600_f21_hw02.py
# PALANI JOHNSON
# A02231136
#########################################
# Output for create_all_nets()
#  and_3_layer_ann.pck:
#      Iterations: 200
#       Threshold: 0.4
#           Shape: (2x3x1)
#         Retries: 0
#  and_4_layer_ann.pck:
#      Iterations: 200
#       Threshold: 0.4
#           Shape: (2x3x3x1)
#         Retries: 2
#  or_3_layer_ann.pck:
#      Iterations: 200
#       Threshold: 0.4
#           Shape: (2x3x1)
#         Retries: 0
#  or_4_layer_ann.pck:
#      Iterations: 200
#       Threshold: 0.4
#           Shape: (2x3x3x1)
#         Retries: 0
#  not_3_layer_ann.pck:
#      Iterations: 200
#       Threshold: 0.4
#           Shape: (1x2x1)
#         Retries: 0
#  not_4_layer_ann.pck:
#      Iterations: 200
#       Threshold: 0.4
#           Shape: (1x2x2x1)
#         Retries: 2
#  xor_3_layer_ann.pck:
#      Iterations: 500
#       Threshold: 0.4
#           Shape: (2x3x1)
#         Retries: 0
#  xor_4_layer_ann.pck:
#      Iterations: 500
#       Threshold: 0.4
#           Shape: (2x3x3x1)
#         Retries: 1
#  bool_3_layer_ann.pck:
#      Iterations: 800
#       Threshold: 0.4
#           Shape: (4x2x1)
#         Retries: 0
#  bool_4_layer_ann.pck:
#      Iterations: 1000
#       Threshold: 0.4
#           Shape: (4x2x2x1)
#         Retries: 1


from numpy.random import normal
from numpy import dot, exp
import pickle
from data import *

# sigmoid function and its derivative.
sigmoidf = lambda i: 1 / (1 + exp(-1 * i))
sigmoidf_prime = lambda i: i * (1 - i)

# feedfoward through a neural net
def feedfoward(inputs, weights):
    activations = [inputs]
    for w in weights:
        activations.append(sigmoidf(activations[-1].dot(w)))
    return activations


# backprop through a neural net
def backprop(activations, weights, truth):
    weights = list(weights)
    cost = truth - activations[-1]
    for i in range(len(weights), 0, -1):
        delta = cost * sigmoidf_prime(activations[i])
        cost = delta.dot(weights[i - 1].T)
        weights[i - 1] += activations[i - 1].T.dot(delta)
    return tuple(weights)


# train a neural net for 1 iteration
def train(inputs, truth, weights):
    return backprop(feedfoward(inputs, weights), weights, truth)


# persists object obj to a file with pickle.dump()
def save(obj, file_name):
    with open(file_name, "wb") as fp:
        pickle.dump(obj, fp)


# restores the object from a file with pickle.load()
def load(file_name):
    with open(file_name, "rb") as fp:
        obj = pickle.load(fp)
    return obj


# neural net inits
build_nn_wmats = lambda mat_dims: tuple(
    [normal(size=(d1, d2)) for d1, d2 in zip(mat_dims[:-1], mat_dims[1:])]
)
build_231_nn = lambda: build_nn_wmats((2, 3, 1))
build_2331_nn = lambda: build_nn_wmats((2, 3, 3, 1))
build_221_nn = lambda: build_nn_wmats((2, 2, 1))
build_838_nn = lambda: build_nn_wmats((8, 3, 8))
build_949_nn = lambda: build_nn_wmats((9, 4, 9))
build_4221_nn = lambda: build_nn_wmats((4, 2, 2, 1))
build_421_nn = lambda: build_nn_wmats((4, 2, 1))
build_121_nn = lambda: build_nn_wmats((1, 2, 1))
build_1221_nn = lambda: build_nn_wmats((1, 2, 2, 1))


def fit_3_layer_nn(input, weights, thresh=0.4, thresh_flag=False):
    output = feedfoward(input, weights)[-1]
    return output > thresh if thresh_flag else output


def fit_4_layer_nn(input, weights, thresh=0.4, thresh_flag=False):
    return fit_3_layer_nn(input, weights, thresh, thresh_flag)


def train_3_layer_nn(iters, input, truth, builder):
    weights = builder()
    for _ in range(iters):
        weights = train(input, truth, weights)
    return weights


def train_4_layer_nn(iters, input, truth, builder):
    return train_3_layer_nn(iters, input, truth, builder)


def create_all_nets():
    for iters, thresh, input, truth, shape, save_to in [
        (200, 0.4, X1, y_and, (2, 3, 1), "data/nets_hw02/and_3_layer_ann.pck"),
        (200, 0.4, X1, y_and, (2, 3, 3, 1), "data/nets_hw02/and_4_layer_ann.pck"),
        (200, 0.4, X1, y_or, (2, 3, 1), "data/nets_hw02/or_3_layer_ann.pck"),
        (200, 0.4, X1, y_or, (2, 3, 3, 1), "data/nets_hw02/or_4_layer_ann.pck"),
        (200, 0.4, X2, y_not, (1, 2, 1), "data/nets_hw02/not_3_layer_ann.pck"),
        (200, 0.4, X2, y_not, (1, 2, 2, 1), "data/nets_hw02/not_4_layer_ann.pck"),
        (500, 0.4, X1, y_xor, (2, 3, 1), "data/nets_hw02/xor_3_layer_ann.pck"),
        (500, 0.4, X1, y_xor, (2, 3, 3, 1), "data/nets_hw02/xor_4_layer_ann.pck"),
        (800, 0.4, X3, bool_exp, (4, 2, 1), "data/nets_hw02/bool_3_layer_ann.pck"),
        (1000, 0.4, X3, bool_exp, (4, 2, 2, 1), "data/nets_hw02/bool_4_layer_ann.pck"),
    ]:
        done = False
        retries = 0
        while not done:
            weights = build_nn_wmats(shape)
            for _ in range(iters):
                weights = train(input, truth, weights)

            done = True

            for i in range(len(input)):
                done = (
                    done
                    and (
                        fit_3_layer_nn(input[i], weights, thresh, thresh_flag=True)
                        == truth[i]
                    ).all()
                )

            if done:
                save(weights, save_to)
                print(f" {save_to[14:]}:")
                print(f"     Iterations: {iters}")
                print(f"      Threshold: {thresh}")
                print(f"          Shape: {str(shape).replace(', ', 'x')}")
                print(f"        Retries: {retries}")
            retries += 1


if __name__ == "__main__":
    create_all_nets()
