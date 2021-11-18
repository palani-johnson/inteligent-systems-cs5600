# /usr/bin/python

from ann import ann
from mnist_loader import load_data_wrapper


train_d, valid_d, test_d = load_data_wrapper()

HLS = [10, 25, 50]
ETA = [0.5, 0.25, 0.125]


def train_1_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10):
    for h in hls:
        for e in eta:
            nn_train([784, h, 10], num_epochs, mini_batch_size, e)


def train_2_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10):
    for h1 in hls:
        for h2 in hls:
            for e in eta:
                nn_train([784, h1, h2, 10], num_epochs, mini_batch_size, e)


def nn_train(shape, num_epochs, mini_batch_size, eta):
    net = ann(shape)
    print(f'*** Training {"X".join([str(l) for l in shape])} ANN with eta={eta}')
    net.mini_batch_sgd(train_d, num_epochs, mini_batch_size, eta, test_data=test_d)
    print(f'*** Training {"X".join([str(l) for l in shape])} ANN DONE...\n')
    return 1


def parallel_train():
    from multiprocessing import Pool

    results = []

    p = Pool(10)
    for h in HLS:
        for e in ETA:
            p.apply_async(
                nn_train,
                args=([784, h, 10], 10, 10, e),
                callback=lambda x: results.append(x),
            )
    p.close()
    p.join()
    print(len(results))


if __name__ == "__main__":
    # train_1_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10)
    # train_2_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10)
    parallel_train()
