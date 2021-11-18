# /usr/bin/python

from ann import *
import numpy as np
import matplotlib.pyplot as plt
from mnist_loader import load_data_wrapper

####################################
# CS5600/6600: F21: HW04
# PALANI JOHNSON
# A02231136
#####################################
# net1.json:
#           Hidden layers: 10
#                     eta: 0.3375
#                  lambda: 2.0
#         Cost (training): 0.5653398800354731
#     Accuracy (training): 46312 / 50000
#       Cost (evaluation): 0.9357279383784438
#   Accuracy (evaluation): 9186 / 10000
#
# net2.json:
#           Hidden layers: 11x10
#                     eta: 0.3375
#                  lambda: 4.0
#         Cost (training): 0.5779308469778776
#     Accuracy (training): 46558 / 50000
#       Cost (evaluation): 1.1385131148611787
#   Accuracy (evaluation): 9278 / 10000
#
# net3.json:
#           Hidden layers: 10x11x11
#                     eta: 0.225
#                  lambda: 4.0
#         Cost (training): 0.6400895564289041
#     Accuracy (training): 45879 / 50000
#       Cost (evaluation): 1.213988847744043
#   Accuracy (evaluation): 9102 / 10000

#### auxiliary functions
def load(filename):
    """
    Load a neural network from the file ``filename``.  Returns an
    instance of ann.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = ann(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


#### plotting costs and accuracies
def plot_costs(eval_costs, train_costs, num_epochs):
    plt.title("Evaluation Cost (EC) and Training Cost (TC)")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    epochs = [i for i in range(num_epochs)]
    plt.plot(epochs, eval_costs, label="EC", c="g")
    plt.plot(epochs, train_costs, label="TC", c="b")
    plt.grid()
    plt.autoscale(tight=True)
    plt.legend(loc="best")
    plt.show()


def plot_accuracies(eval_accs, train_accs, num_epochs):
    plt.title("Evaluation Acc (EA) and Training Acc (TC)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    epochs = [i for i in range(num_epochs)]
    plt.plot(epochs, eval_accs, label="EA", c="g")
    plt.plot(epochs, train_accs, label="TA", c="b")
    plt.grid()
    plt.autoscale(tight=True)
    plt.legend(loc="best")
    plt.show()


## num_nodes -> (eval_cost, eval_acc, train_cost, train_acc)
## use this function to compute the eval_acc and min_cost.
def collect_1_hidden_layer_net_stats(
    lower_num_hidden_nodes,
    upper_num_hidden_nodes,
    cost_function,
    num_epochs,
    mbs,
    eta,
    lmbda,
    train_data,
    eval_data,
):
    incr = lambda: range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1)
    return collect_n_hidden_layer_net_stats(
        lambda x: x[0],
        [[i] for i in incr()],
        cost_function,
        num_epochs,
        mbs,
        eta,
        lmbda,
        train_data,
        eval_data,
    )


def collect_2_hidden_layer_net_stats(
    lower_num_hidden_nodes,
    upper_num_hidden_nodes,
    cost_function,
    num_epochs,
    mbs,
    eta,
    lmbda,
    train_data,
    eval_data,
):
    incr = lambda: range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1)
    return collect_n_hidden_layer_net_stats(
        lambda x: "_".join([str(i) for i in x]),
        [[i, j] for i in incr() for j in incr()],
        cost_function,
        num_epochs,
        mbs,
        eta,
        lmbda,
        train_data,
        eval_data,
    )


def collect_3_hidden_layer_net_stats(
    lower_num_hidden_nodes,
    upper_num_hidden_nodes,
    cost_function,
    num_epochs,
    mbs,
    eta,
    lmbda,
    train_data,
    eval_data,
):
    incr = lambda: range(lower_num_hidden_nodes, upper_num_hidden_nodes + 1)
    return collect_n_hidden_layer_net_stats(
        lambda x: "_".join([str(i) for i in x]),
        [[i, j, k] for i in incr() for j in incr() for k in incr()],
        cost_function,
        num_epochs,
        mbs,
        eta,
        lmbda,
        train_data,
        eval_data,
    )


def collect_n_hidden_layer_net_stats(
    dict_formatter,
    hidden_layers,
    cost_function,
    num_epochs,
    mbs,
    eta,
    lmbda,
    train_data,
    eval_data,
):
    results = {}
    for hidden_layer in hidden_layers:
        layers = [784] + hidden_layer + [10]
        net = ann(layers, cost=cost_function)
        results[dict_formatter(hidden_layer)] = net.mini_batch_sgd(
            train_data,
            num_epochs,
            mbs,
            eta,
            lmbda,
            eval_data,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True,
        )

    return results


if __name__ == "__main__":
    train_d, valid_d, test_d = load_data_wrapper()

    # Prints the results from the stats functions to assist in
    # finding a good structure
    def find_structure(stats_function, l, u, lmbda=0.5, eta=0.25):
        data = stats_function(
            l, u, CrossEntropyCost, 3, 10, lmbda, eta, train_d, test_d
        )

        for k, v in data.items():
            v = "\n".join([f"    {w}" for w in v])
            print(f"*** {k} ***\n{v}")
        print("************\n")

    # find_structure(collect_1_hidden_layer_net_stats, 10, 11)
    # find_structure(collect_2_hidden_layer_net_stats, 10, 11)
    # find_structure(collect_3_hidden_layer_net_stats, 10, 11)

    best_1_hidden = [10]
    best_2_hidden = [11, 10]
    best_3_hidden = [10, 11, 11]

    def find_eta(hidden_layers, lmbda=0.0, eta_start=0.1, eta_mult=1.5, n=10):
        layers = [784] + hidden_layers + [10]
        eta = eta_start
        for _ in range(n):
            print(f"*** eta = {eta} ***")
            net = ann(layers, cost=CrossEntropyCost)
            net.mini_batch_sgd(
                train_d,
                4,
                10,
                eta,
                lmbda,
                test_d,
                monitor_evaluation_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_training_cost=True,
                monitor_training_accuracy=True,
            )
            eta *= eta_mult

    # print("*********************** 1 hidden *************************\n")
    # find_eta(best_1_hidden)
    # print("*********************** 2 hidden *************************\n")
    # find_eta(best_2_hidden)
    # print("*********************** 3 hidden *************************\n")
    # find_eta(best_3_hidden)

    best_1_eta = 0.3375
    best_2_eta = 0.25
    best_3_eta = 0.3375

    def find_lambda(hidden_layers, eta, lambda_start=0.25, lambda_mult=2, n=10):
        layers = [784] + hidden_layers + [10]
        lmbda = lambda_start
        for _ in range(n):
            print(f"*** lambda = {lmbda} ***")
            net = ann(layers, cost=CrossEntropyCost)
            net.mini_batch_sgd(
                train_d,
                4,
                10,
                eta,
                lmbda,
                test_d,
                monitor_evaluation_cost=True,
                monitor_evaluation_accuracy=True,
                monitor_training_cost=True,
                monitor_training_accuracy=True,
            )
            lmbda *= lambda_mult

    # print("*********************** 1 hidden *************************\n")
    # find_lambda(best_1_hidden, best_1_eta)
    # print("*********************** 2 hidden *************************\n")
    # find_lambda(best_2_hidden, best_2_eta)
    # print("*********************** 3 hidden *************************\n")
    # find_lambda(best_3_hidden, best_3_eta)

    best_1_lambda = 2.0
    best_2_lambda = 4.0
    best_3_lambda = 4.0

    # print("*********************** 1 hidden *************************\n")
    # find_eta(best_1_hidden, best_1_lambda)
    # print("*********************** 2 hidden *************************\n")
    # find_eta(best_2_hidden, best_2_lambda)
    # print("*********************** 3 hidden *************************\n")
    # find_eta(best_3_hidden, best_3_lambda)

    best_1_eta = 0.3375
    best_2_eta = 0.3375
    best_3_eta = 0.225

    def train_and_save(hidden_layers, eta, lmbda, name):
        layers = [784] + hidden_layers + [10]
        net = ann(layers, cost=CrossEntropyCost)
        net.mini_batch_sgd(
            train_d,
            30,
            10,
            eta,
            lmbda,
            test_d,
            monitor_evaluation_cost=True,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True,
        )
        net.save(f"data/nets_hw04/{name}")

    # print("*********************** 1 hidden *************************\n")
    # train_and_save(best_1_hidden, best_1_eta, best_1_lambda, "net1.json")
    # print("*********************** 2 hidden *************************\n")
    # train_and_save(best_2_hidden, best_2_eta, best_2_lambda, "net2.json")
    print("*********************** 3 hidden *************************\n")
    train_and_save(best_3_hidden, best_3_eta, best_3_lambda, "net3.json")
