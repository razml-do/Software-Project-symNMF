import sys
import math
import kmeans
import symnmf
from sklearn.metrics import silhouette_score

def read_matrix_from_txt(filename):
    """Read a comma-separated matrix from a txt file."""
    data = []
    with open(filename, 'r') as f:
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
    return math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, q)))

def get_cluster_kmeans(p, centroids):
    cluster_index = 0
    min_dist_from_centroid = float('inf')
    for i in range(len(centroids)):
        if min_dist_from_centroid > euclidean_distance(p, centroids[i]):
            min_dist_from_centroid = euclidean_distance(p, centroids[i])
            cluster_index = i
            
    return cluster_index

def get_cluster_symnmf(H_p):
    cluster_index = 0
    for i in range(len(H_p)):
        if H_p[cluster_index] < H_p[i]:
            cluster_index = i
            
    return cluster_index

def compute_symnmf_labels(X, symnmf_mat):
    labels = []
    for i in range(len(X)):
        labels.append(get_cluster_symnmf(symnmf_mat[i]))
    return labels

def compute_kmeans_labels(X, kmeans_mat):
    labels = []
    for i in range(len(X)):
        labels.append(get_cluster_kmeans(X[i], kmeans_mat))
    return labels
def main():
    k = int(sys.argv[1])
    filename = sys.argv[2]
    X = read_matrix_from_txt(filename)
    kmeans_mat = kmeans.analysis_kmeans(X, k)
    symnmf_mat = symnmf.symnmf(X, k)

    print("KMeans:")
    symnmf.print_matrix(kmeans_mat)        
    print("SymNMF:")
    symnmf.print_matrix(symnmf_mat)
    symnmf_labels = compute_symnmf_labels(X, symnmf_mat)
    kmeans_labels = compute_kmeans_labels(X, kmeans_mat)
    print(symnmf_labels)
    print(kmeans_labels)
    symnmf_sil = silhouette_score(X, symnmf_labels)
    kmeans_sil = silhouette_score(X, kmeans_labels)

    print("nmf: {:.4f}".format(symnmf_sil))
    print("kmeans: {:.4f}".format(kmeans_sil))
        

if __name__ == "__main__":
    main()
