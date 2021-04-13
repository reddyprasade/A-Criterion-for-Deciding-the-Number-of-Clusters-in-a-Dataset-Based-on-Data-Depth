import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


def get_iris_X_num() -> (np.array, int):
    """
    Return the iris dataset with the true number of clusters.
    Returns: feature matrix, label matrix, number of classes

    """
    iris_X, iris_y = datasets.load_iris(return_X_y=True)  # (150, 4), (150, )
    num_clusters = int(np.unique(iris_y).shape[0])
    return iris_X, num_clusters


def get_birch() -> tuple([np.array, int]):
    """
    Get the BIRCH dataset from http://cs.joensuu.fi/
    Args:
        url: specific url of the dataset

    Returns: Dataset matrix (n, m) where m is number of features

    """
    X = np.loadtxt(fname='datasets/birch3')
    return X, 100


def get_dim_128() -> tuple([np.array, int]):
    """
    Get the dim128 dataset from http://cs.joensuu.fi/
    Args:
        url: specific url of the dataset

    Returns: Dataset matrix (n, m) where m is number of features

    """
    X = np.loadtxt(fname='datasets/dim128')
    return X, 16
