import numpy as np

W = np.random.uniform(low=-0.05, high=0.05, size=(12, 6))
W2 = np.random.normal(0.0, 0.5, size=(12, 6))
W3 = np.ones((12, 6))
#print(W)
#print()
#print(W2)
print(W3)


def test(layers, alpha=0.1):
    W = []
    layers = layers
    alpha = alpha

    for i in np.arange(0, len(layers) - 2):
        w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
        W.append(w / np.sqrt(layers[i]))

    w = np.random.randn(layers[-2] + 1, layers[-1])
    W.append(w / np.sqrt(layers[-2]))
    print()
    print(W)


test([8, 4, 2])


