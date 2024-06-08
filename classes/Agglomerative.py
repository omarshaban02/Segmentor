import numpy as np


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def clusters_distance_2(cluster1, cluster2):
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)


class AgglomerativeClustering:

    def __init__(self, k=2, initial_k=25):
        self.k = k
        self.initial_k = initial_k

    def initial_clusters(self, points):
        initial_groups = {}
        d = int(256 / (self.initial_k))
        for i in range(self.initial_k):
            j = i * d
            initial_groups[(j, j, j)] = []
        for i, p in enumerate(points):
            go = min(initial_groups.keys(), key=lambda c: euclidean_distance(p, c))
            initial_groups[go].append(p)
        return [g for g in initial_groups.values() if len(g) > 0]

    def fit(self, points):

        # initially, assign each point to a distinct cluster
        self.clusters_list = self.initial_clusters(points)

        while len(self.clusters_list) > self.k:
            # Find the closest (most similar) pair of clusters
            cluster1, cluster2 = min(
                [(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                key=lambda c: clusters_distance_2(c[0], c[1]))

            # Remove the two clusters from the clusters list
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]

            # Merge the two clusters
            merged_cluster = cluster1 + cluster2

            self.clusters_list.append(merged_cluster)

        # final cluster dictionary
        self.cluster = {}

        # Iterate over pixels in each cluster and give it the number of it is cluster
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num

        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)

    def predict_cluster(self, point):
        return self.cluster[tuple(point)]

    # take the pixel and return the new value of this pixel, first we know what it is cluster number and return
    # the average of this cluster as the new value of this pixel
    def predict_center(self, point):
        point_cluster_number = self.predict_cluster(point)
        center = self.centers[point_cluster_number]
        return center
