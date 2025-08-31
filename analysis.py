import sys
import math
import kmeans
import symnmf
from sklearn.metrics import silhouette_score

def read_matrix_from_txt(file_name):
    """
    Read a comma-separated matrix from a txt file.
    :param file_name: file to read
    """
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = [float(x) for x in line.split(',')]
                except Exception:
                    raise ValueError("Invalid data point")
                data.append(row)
                
    return data

def euclidean_distance(p, q):
    """
    Calculate euclidean distance between two points.

    :param p: first point
    :param q: second point
    """
    return math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))

def get_cluster_kmeans(p, centroids):
    """
    Get cluster of point from centroids matrix
    produced by kmeans algorithm.

    :param p: point to calculate it's cluster
    :param centroids: centroids matrix from kmeans algorithm
    """
    cluster_index = 0
    min_dist_from_centroid = float('inf')
    for i in range(len(centroids)):
        point_dist = euclidean_distance(p, centroids[i])
        if min_dist_from_centroid > point_dist:
            min_dist_from_centroid = point_dist
            cluster_index = i
            
    return cluster_index

def get_cluster_symnmf(H_p):
    """
    Get cluster of point from symnmf matrix
    produced by symnmf algorithm.

    :param H_p: point (from symnmf matrix!)
    """
    cluster_index = 0
    for i in range(len(H_p)):
        if H_p[cluster_index] < H_p[i]:
            cluster_index = i
            
    return cluster_index

def compute_symnmf_labels(X, symnmf_mat):
    """
    Compute the 1D array 'labels' that represents the cluster
    each point belongs to (index i is the point and labels[i] is the cluster)
    according to symnmf matrix
    
    :param X: input matrix
    :param symnmf_mat: matrix produced by symnmf algorithm
    """
    labels = []
    for i in range(len(X)):
        labels.append(get_cluster_symnmf(symnmf_mat[i]))
    return labels

def compute_kmeans_labels(X, kmeans_mat):
    """
    Compute the 1D array 'labels' that represents the cluster
    each point belongs to (index i is the point and labels[i] is the cluster)
    according to kmeans matrix
    
    :param X: input matrix
    :param kmeans_mat: matrix produced by kmeans algorithm
    """
    labels = []
    for i in range(len(X)):
        labels.append(get_cluster_kmeans(X[i], kmeans_mat))
    return labels

def main():
    """
    analysis.py computes 
    kmeans centroid & symnmf association matrices from input points
    and calculates the silhouette_score of both

    :sys.argv[1]: required number of clusters
    :sys.argv[2]: input points (.txt file, 2D matrix of nxd 
                  when n is the number of points and d is their dimension)
    """
    try:
        k = int(sys.argv[1])
        filename = sys.argv[2]
        X = read_matrix_from_txt(filename)
        kmeans_mat = kmeans.analysis_kmeans(X, k)
        symnmf_mat = symnmf.symnmf(X, k)
        symnmf_labels = compute_symnmf_labels(X, symnmf_mat)
        kmeans_labels = compute_kmeans_labels(X, kmeans_mat)
        symnmf_sil = silhouette_score(X, symnmf_labels)
        kmeans_sil = silhouette_score(X, kmeans_labels)
        print("nmf: {:.4f}".format(symnmf_sil))
        print("kmeans: {:.4f}".format(kmeans_sil))
    except:
        print("An Error Has Occurred")
        sys.exit(1)
    
        

if __name__ == "__main__":
    main()
