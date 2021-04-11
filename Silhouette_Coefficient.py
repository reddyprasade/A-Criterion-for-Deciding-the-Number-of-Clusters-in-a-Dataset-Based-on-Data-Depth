"""
Silhouette Coefficient


The Silhouette Coefficient for a point i is defined as follows:
where b(i) is the smallest average distance of point i to all points in any other cluster and a(i) is the average distance of i from all other points in its cluster. 
For example, if we have only 3 clusters A,B and C and i belongs to cluster C, then b(i) is calculated by measuring the average distance of i from every point in cluster A, the average distance of i from every point in cluster B and taking the smallest resulting value. 
The Silhouette Coefficient for the dataset is the average of the Silhouette Coefficient of individual points.
The Silhouette Coefficient tells us if individual points are correctly assigned to their clusters. 

We can use the following thumb rules while using Silhouette Coefficient:
1. S(i) close to 0 means that the point is between two clusters
2. If it is closer to -1, then we would be better off assigning it to the other clusters
3. If S(i) is close to 1, then the point belongs to the ‘correct’ cluster

"""
from sklearn import datasets
from sklearn.cluster import KMeans


iris = datasets.load_iris()
cluster_df = iris.data

# Silhouette Score for K means
# Import ElbowVisualizer
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
# k is range of number of clusters.
visualizer = KElbowVisualizer(model, k=(2,30),metric='silhouette', timings= True)
visualizer.fit(cluster_df)        # Fit the data to the visualizer
visualizer.show() 