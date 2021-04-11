"""

Davies-Bouldin Index


The Davies-Bouldin (DB) Index is defined as:

where n is the count of clusters and σi is the average distance of all points in cluster i from the cluster centre ci.
Like silhouette coefficient and Calinski-Harabasz index, the DB index captures both the separation and compactness of the clusters.
This is due to the fact that the measure’s ‘max’ statement repeatedly selects the values where the average point is farthest away from its center, and where the centers are closest together. 
But unlike silhouette coefficient and Calinski-Harabasz index, as DB index falls, the clustering improves.
"""

# Davies Bouldin score for K means
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets


iris = datasets.load_iris()
cluster_df = iris.data

def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding Davies Bouldin for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the Davies Bouldin score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = KMeans(n_clusters=center)
# Then fit the model to your data using the fit method
    model = kmeans.fit_predict(cluster_df)
    
    # Calculate Davies Bouldin score
    score = davies_bouldin_score(cluster_df, model)
    
    return score
scores = []
centers = list(range(2,30))
for center in centers:
    scores.append(get_kmeans_score(cluster_df, center))

print(" Davies Bouldin score for the kmeans model fit to the data",scores)

plt.plot(centers, scores, linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('Davies Bouldin score');
plt.title('Davies Bouldin score vs. K');
plt.show()