from data_parser import get_iris_X_num, get_birch, get_dim_128
from DeD_alg import depth_difference_algorithm
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

X, n = get_iris_X_num()
#X, n = get_dim_128()

estimated = []
for _ in range(1):
    estimated.append(depth_difference_algorithm(shuffle(X), k_min=2, k_max=23))

print(estimated[0])
plt.hist(estimated)
plt.show()
