"""

Calinski-Harabasz Index

The Calinski-Harabasz Index is based on the idea that clusters that are (1) themselves very compact and (2) well-spaced from each other are good clusters. 
The index is calculated by dividing the variance of the sums of squares of the distances of individual objects to their cluster center by the sum of squares of the distance between the cluster centers. 
Higher the Calinski-Harabasz Index value, better the clustering model. The formula for Calinski-Harabasz Index is defined as:

where k is the number of clusters, n is the number of records in data, BCSM (between cluster scatter matrix) calculates separation between clusters and WCSM (within cluster scatter matrix) calculates compactness within clusters.
KElbowVisualizer function is able to calculate Calinski-Harabasz Index as well
"""

# Calinski Harabasz Score for K means
# Import ElbowVisualizer
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn import datasets


iris = datasets.load_iris()
cluster_df = iris.data
model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30),metric='calinski_harabasz', timings= True)
visualizer.fit(cluster_df)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure