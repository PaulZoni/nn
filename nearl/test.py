import numpy as np

x = np.array([-3.44, 1.16, 3.91])

print(1 + np.exp(x))
print(1.0 / (1 + np.exp(-x)))

S = np.exp(x)

print(S[2] / S.sum())










