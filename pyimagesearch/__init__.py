from scipy.io import loadmat
import numpy

mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
mnist_path = "./mnist-original.mat"


def load_MNIST(path):
    mnist_raw = loadmat(path)

    mnist = {
        "data": numpy.array(mnist_raw["data"].T),
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    return mnist.get('data')


def load_target_MNIST(path):
    mnist_raw = loadmat(path)

    mnist = {
        "data": numpy.array(mnist_raw["data"].T),
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    return mnist.get('target')


def MNIST(path):
    mnist_raw = loadmat(path)

    mnist = {
        "data": numpy.array(mnist_raw["data"].T),
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    return mnist.get('target')


'''import tensorflow as tf


hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))'''

#print(MNIST(mnist_path)[1])
