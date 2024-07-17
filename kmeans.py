import math
import random

class Kmeans:
    def __init__(self, k):
        self.k = k
        self.centeriod_init = 'random'
        self.max_iters = 100
        self.centers = []
    def euclidean_dist(self, x, y):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))

    def initialise_centroids(self, init_type, k, data):
          if init_type == 'random':
            indices = list(range(len(data)))
            random.shuffle(indices)
            return [data[i] for i in indices[:k]]

    def fit(self, data):
        m = len(data)
        cluster_assignments = [[0, 0] for _ in range(m)]

        cents = self.initialise_centroids(self.centeriod_init, self.k, data)

        cents_orig = [list(centroid) for centroid in cents]
        changed = True
        num_iter = 0

        while changed and num_iter < self.max_iters:
            changed = False
            for i in range(m):
                min_dist = float('inf')
                min_index = -1
                # calculate distance
                for j in range(self.k):
                    dist_ji = self.euclidean_dist(cents[j], data[i])
                    if dist_ji < min_dist:
                        min_dist = dist_ji
                        min_index = j
                    if cluster_assignments[i][0] != min_index:
                        changed = True

                cluster_assignments[i] = [min_index, min_dist ** 2]


            for cent in range(self.k):
                points = [data[i] for i, assignment in enumerate(cluster_assignments) if assignment[0] == cent]
                cents[cent] = [sum(values) / len(values) for values in zip(*points)]

            num_iter += 1


        self.centers = cents
        return cents, cluster_assignments, num_iter, cents_orig

    def predict(self,data):
        m = len(data)
        predictions = [[0, 0] for _ in range(m)]
        for i in range(m):
            min_dist = float('inf')
            min_index = -1
            for j in range(self.k):
                dist_ji = self.euclidean_dist(self.centers[j], data[i])
                if dist_ji < min_dist:
                    min_dist = dist_ji
                    min_index = j
            predictions[i] = [min_index, min_dist **2]

        return predictions    
