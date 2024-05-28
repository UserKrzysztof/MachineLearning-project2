import numpy as np
from scipy.spatial import distance
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score


def clustering_score(X, model, score_fun):
    labels = model.fit_predict(X)
    return  score_fun(X, labels)


#I dunno if we need this when we have other metrics
def mean_center_distance(model, X):
    labels = model.fit_predict(X)
    clusters = np.unique(labels)
    inclust_dist_list = []
    
    for cluster in clusters:
        cluster_indices = np.where(labels == cluster)[0]
        cluster_points = X[cluster_indices]
        cluster_center = np.mean(cluster_points, axis=0, keepdims=True)
        inclust_dist = np.mean(distance.cdist(cluster_points, [cluster_center]))
        inclust_dist_list.append(inclust_dist)
    
    return - np.mean(inclust_dist_list)


#This index signifies the average ‘similarity’ between clusters, where the similarity 
#a measure that compares the distance between clusters with the size of the clusters themselves.
#Zero is the lowest possible score. Values closer to zero indicate a better partition.
def davies_bouldin(model, X):
    labels = model.fit_predict(X)
    return 1 / davies_bouldin_score(X, labels)

def silhouette(model, X):
    labels = model.fit_predict(X)
    return silhouette_score(X, labels)

def calinski_harabasz(model, X):    
    labels = model.fit_predict(X)
    return calinski_harabasz_score(X, labels)

#I dunno if we need this when we have other metrics
def mean_distrance_in_clusters(model,X):
    labels = model.fit_predict(X)
    clusters = set(labels)
    inclust_distance_list = []
    for cluster in clusters:
        cluster_indices = np.where(labels == cluster)[0]
        inclust_distance = np.mean(distance.pdist(X[cluster_indices]))
        inclust_distance_list.append(inclust_distance)
    return np.mean(inclust_distance_list)


#I dunno if we need this when we have other metrics
def min_distance_between_clusters(model, X):
    labels = model.fit_predict(X)
    clusters = set(labels)
    global_min_dist = np.inf
    for cluster1 in clusters:
        cluster1_indices = np.where(labels == cluster1)[0]
        for cluster2 in clusters:
            if cluster1 != cluster2:
                cluster2_indices = np.where(labels == cluster2)[0]
                min_dist = np.min(distance.cdist(X[cluster1_indices], X[cluster2_indices]))
                global_min_dist = np.min([global_min_dist, min_dist])
    return global_min_dist