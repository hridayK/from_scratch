import numpy as np
from sympy import centroid

np.random.seed(42)

def euclidian_distance(x1,x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return np.sqrt(np.sum((x1-x2)**2))

class Kmeans:
    def __init__(self, K=5, max_iter=5, plot_steps=False):
        self.K = K
        self.max_iter = max_iter
        self.plot_steps = plot_steps
        #initializing K empty lists i.e. K empty clusters
        self.clusters = [[] for _ in range(self.K)]
        self.centorid = []

    def predict(self,X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize centroids
        random_sample_idx = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idx]
        # optimization
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self.create_clusters(self.centroids)
            
            # update centroids
            old_centroids = self.centroids
            self.centroids = self.get_centroids()

    def create_clusters(self, centroids):
        # creating K empty clusters
        clusters = [[] for _ in range(self.K)]

        # making clusters
        for idx,sample in enumerate(self.X):
            centroid_idx = self.closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
            return clusters
        
    def closest_centroid(self, sample, centroids):
        distances = [euclidian_distance(sample, point) for point in centroids]
        return np.argmin(distances)

    def get_centroids(self, centroids):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(cluster):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        
        return centroids