import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances


def depth_in_cluster(cluster: np.array) -> np.array:
    """
    Calculate the data depth of each point in a cluster Ck ("cluster")
    as defined in Patil et. al.
    Args:
        cluster: One cluster from the partition to calculate the metric within.

    Returns: Array with data depth of each point.

    """
    if(cluster.shape[1] == 1):
        return 0
    sigma = np.cov(cluster)  # Covariance matrix
    mu = np.mean(cluster, axis=1)  # mean vector
    depths = []
    for i in range(cluster.shape[1]):
        # mahalanobis_result = mahalanobis(cluster[i], mu, np.linalg.inv(sigma))
        # mahalanobis_result = (cluster.T[i] - mu).T.dot(np.linalg.pinv(sigma)).dot(cluster.T[i] - mu)
        mahalanobis_result = cosine_distances(cluster.T[i].reshape(-1, 1), mu.reshape(-1, 1))
        depth = 1 / (1 + mahalanobis_result)
        depths.append(depth)

    return np.array(depths)


def average_depth_difference_in_cluster(Dk: np.array, DMk: np.array) -> np.array:
    """
    For a given cluster, calculate the average depth difference between the depth of a point and the depth-median
    over all points in the cluster, as presented in Patil et. al.
    Args:
        Dk: An array of depth of each point within a cluster.
        DMk: The depth median of the cluster.

    Returns: average depth difference of all points in the cluster (1 / n) * sigma_i (|Dk_i - DMk|)

    """
    abs_differences = np.abs(Dk - DMk)
    avg_depth_difference = np.mean(abs_differences)

    return avg_depth_difference


def depth_difference_algorithm(X: np.array, k_min: int = 2, k_max: int = 20) -> int:
    """
    Implementation of the data depth algorithm as presented in Patil et. al.
    return the estimated number of clusters in the given dataset.
    Args:
        X: dataset of points from an arbitrary dimension. X = {x1, x2 ... xn}
            It's a matrix from the shape (n, m), m is the number of features.
        k_min: minimum number of clusters in the data
        k_max: maximum number of clusters in the data

    Returns: k, the number of estimated clusters in the dataset.

    """
    D = depth_in_cluster(X.T)
    DM = np.max(D)
    delta = average_depth_difference_in_cluster(D, DM)  # Average depth difference of whole dataset
    n = X.shape[0]
    DeD = []
    for k in tqdm(range(k_min, k_max)):
        partition_size = int(np.floor(n / k))
        start, end = 0, 0
        Dk = []
        delta_k = []
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        for j in range(k):
            # start = end
            # end = start + partition_size
            Dk.append(depth_in_cluster(X[kmeans.labels_ == j].T))
            DMk = np.max(Dk[-1])
            delta_k.append(average_depth_difference_in_cluster(np.array(Dk[-1]), DMk))

        DW = np.mean(delta_k)
        DB = delta - DW
        DeD.append(DW - DB)

    return int(np.argmax(DeD)) + k_min
