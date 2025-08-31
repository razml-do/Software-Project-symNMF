import sys
import math

EPSILON = 1e-4
ITER_MAX = 300

def euclidean_distance(p, q):
    """
    Calculate euclidean distance between two points.

    :param p: first point
    :param q: second point
    """
    return math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))

def centroids_converged(prev, current):
    """
    Returns true if distance between current and previous point
    is smaller than epsilon (Δu < ε)

    :param prev: previous point
    :param current: current point
    """
    return all(euclidean_distance(p, c) < EPSILON for p, c in zip(prev, current))

def initialize_centroids(X, k):
    """
    Returns initialized centroids.

    :param data: input data
    :param k: required number of clusters
    """
    return [list(X[i]) for i in range(k)]

def kmeans(X, k):
    """
    Main kmeans algorithm.

    :param X: input data
    :param k: required number of clusters
    """
    N = len(X)
    d = len(X[0])
    if k <= 1 or k >= N:
        raise ValueError("k out of valid range")
    
    centroids = initialize_centroids(X, k)
    prev_centroids = [[0.0] * d for _ in range(k)]
    for _ in range(ITER_MAX):
        clusters = [[] for _ in range(k)]

        # Assignment step
        for idx, point in enumerate(X):
            distances = [euclidean_distance(point, c) for c in centroids]
            closest = distances.index(min(distances))
            clusters[closest].append(idx)

        # Update step
        for i, cluster in enumerate(clusters):
            if cluster:  # avoid division by zero
                new_centroid = [0.0] * d
                for idx in cluster:
                    for j in range(d):
                        new_centroid[j] += X[idx][j]
                centroids[i] = [val / len(cluster) for val in new_centroid]

        if centroids_converged(prev_centroids, centroids):
            break
        
        prev_centroids = [list(c) for c in centroids]

    return centroids

def analysis_kmeans(X, k):
    """
    Main kmeans algorithm for analysis.py file.

    :param X: input data
    :param k: required number of clusters
    """
    try:
        centroids = kmeans(X, k)
        centroids = [[round(val, 4) for val in row] for row in centroids]
        return centroids
    except:
        sys.exit(1)