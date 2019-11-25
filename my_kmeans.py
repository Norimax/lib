import numpy as np


class MyKMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers_log = []

    def init_cluster_ids(self, X):
        self.cluster_ids = np.array(
            [np.random.randint(self.n_clusters) for _ in range(len(X))])

    def update_centers(self, X):
        self.centers = np.array([
            X[np.where(self.cluster_ids == i)].T.mean(1)
            for i in range(self.n_clusters)])
        self.centers_log.append(self.centers)

    def update_cluster_ids(self, X):
        self.cluster_ids = np.array([
            np.argmin([np.linalg.norm(x - c) for c in self.centers])
            for x in X])

    def fit(self, X):
        self.init_cluster_ids(X)
        for _ in range(self.max_iter):
            self.update_centers(X)
            cluster_ids_before = self.cluster_ids
            self.update_cluster_ids(X)
            if (cluster_ids_before != self.cluster_ids).sum() == 0:
                break