"""
It is the most popular method for determining the optimal number of clusters. 
The method is based on calculating the Within-Cluster-Sum of Squared Errors (WSS) for different number of clusters (k) and selecting the k for which change in WSS first starts to diminish.
The idea behind the elbow method is that the explained variation changes rapidly for a small number of clusters and then it slows down leading to an elbow formation in the curve. 
The elbow point is the number of clusters we can use for our clustering algorithm.
"""
# pip install -U yellowbrick
# Elbow Method for K means
# Import ElbowVisualizer

from yellowbrick.cluster import KElbowVisualizer
from sklearn import datasets
from sklearn.cluster import KMeans


iris = datasets.load_iris()
cluster_df = iris.data


model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30), timings= True)
visualizer.fit(cluster_df)        # Fit data to visualizer
visualizer.show()        # Finalize and render figure