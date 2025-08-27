import sys
import numpy as np

def gaussian_similarity_matrix(X, sigma=1.0):
    """
    Compute Gaussian (RBF) similarity matrix.
    
    Parameters:
        X (ndarray): Data matrix of shape (n_samples, n_features)
        sigma (float): Bandwidth parameter for the Gaussian kernel.
        
    Returns:
        K (ndarray): Similarity matrix of shape (n_samples, n_samples)
    """
    # Compute pairwise squared Euclidean distances
    sq_norms = np.sum(X**2, axis=1)
    dists = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(X, X.T)

    # Compute Gaussian kernel
    K = np.exp(-dists / (2 * sigma**2))
    return K

if "gen" == sys.argv[1]:
    # Parameters
    # num_matrices = 5        # How many random datasets
    # sizes = [5, 10, 20, 50, 100]  # Number of points in each dataset
    features = 3            # Dimensionality of points
    sigma = 1.0             # Bandwidth

    # for n in sizes:
    
    n = int(sys.argv[2])
    X = np.random.randn(n, features)  # random Gaussian data
    K = gaussian_similarity_matrix(X, sigma)
    print(f"\nMatrix size: {n}x{n}")
    print(f"Mean similarity: {K.mean():.4f}, Min: {K.min():.4f}, Max: {K.max():.4f}")
    
    #print(X)
    
    for r in range(len(X)):
        print(",".join([f"{k:.4f}" for k in X[r]]))
    
    #print(K)  # Optional: print full matrix for small n
    
elif "sol" == sys.argv[1]: 
    # Example usage
    # X = np.array([[1,2,3,4,5], [5,4,3,2,1], [1,2,3,4,5]])
    #f = sys.argv[1]
    X = [[float(num) for num in ln.strip().split(',')] for ln in open("input_1.txt").readlines()]
    X = np.array(X)
    K = gaussian_similarity_matrix(X, sigma=1.0)
    print(K)
    
else:
    print("use `gen` or `solve`.")