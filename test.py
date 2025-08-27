import numpy as np
import subprocess
import sys
import symnmfmodule  # type: ignore

# ----------------------------
# Global settings
# ----------------------------
k = 3  # number of clusters (change once here)
np.random.seed(1234)  # for reproducibility

# Small dataset for testing
X_small = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [4.0, 5.0],
    [5.0, 4.0]
])

# ----------------------------
# Helper functions
# ----------------------------
def save_csv(matrix, filename):
    """Save matrix as comma-separated, 4 decimal places"""
    np.savetxt(filename, matrix, delimiter=",", fmt="%.4f")

def print_matrix(matrix, name="Matrix"):
    print(f"\n{name}:")
    for row in matrix:
        print(" ".join(f"{v:.4f}" for v in row))

# ----------------------------
# Test core matrix functions
# ----------------------------
def test_core_functions():
    save_csv(X_small, "X.csv")

    print("--- sym(X.csv) ---")
    W = symnmfmodule.sym("X.csv")
    print_matrix(W, "sym")

    print("--- ddg(X.csv) ---")
    D = symnmfmodule.ddg("X.csv")
    print_matrix(D, "ddg")

    print("--- norm(X.csv) ---")
    N = symnmfmodule.norm("X.csv")
    print_matrix(N, "norm")

# ----------------------------
# Test SymNMF algorithm
# ----------------------------
def test_symnmf_algorithm():
    save_csv(X_small, "data.csv")
    H_init = np.random.rand(X_small.shape[0], k)
    W_matrix = symnmfmodule.sym("data.csv")
    save_csv(H_init, "H.csv")
    save_csv(W_matrix, "W.csv")

    H_final = symnmfmodule.symnmf("data.csv", H_csv="H.csv", W_csv="W.csv", k=k)
    print("--- symnmf output ---")
    print_matrix(H_final, "H_final")

# ----------------------------
# Test K-means
# ----------------------------
def test_kmeans():
    save_csv(X_small, "data.csv")
    proc = subprocess.run(
        [sys.executable, "kmeans.py", str(k)],
        input="\n".join(",".join(f"{v:.4f}" for v in row) for row in X_small),
        text=True,
        capture_output=True
    )
    assert proc.returncode == 0, "kmeans.py failed"
    print("--- K-means centroids ---")
    print(proc.stdout.strip())

# ----------------------------
# Test analysis.py
# ----------------------------
def test_analysis_pipeline():
    save_csv(X_small, "data.txt")
    proc = subprocess.run(
        [sys.executable, "analysis.py", str(k), "data.txt"],
        text=True,
        capture_output=True
    )

    print("\n--- analysis.py stdout ---")
    print(proc.stdout.strip())
    print("\n--- analysis.py stderr ---")
    print(proc.stderr.strip())

    if proc.returncode != 0:
        raise RuntimeError(f"analysis.py failed with exit code {proc.returncode}")
    else:
        print("\n--- analysis.py finished successfully ---")

# ----------------------------
# Test small synthetic dataset (stable)
# ----------------------------
def test_small_dataset_stable():
    n_points = 25
    dim = 2

    # Generate small clusters
    cluster_size = n_points // k
    cluster_centers = np.array([[2, 2], [4, 4], [6, 2]])
    X = []
    for center in cluster_centers:
        points = center + np.random.uniform(-0.5, 0.5, size=(cluster_size, dim))
        X.append(points)
    X = np.vstack(X)
    save_csv(X, "small_data.csv")

    # Build similarity matrix
    n = X.shape[0]
    pairwise_dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    sigma = max(np.mean(pairwise_dists), 1e-4)
    W = np.exp(-pairwise_dists**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0.0)
    save_csv(W, "W_small.csv")

    # H_init scaled appropriately
    H_init = np.random.uniform(0, 2*np.sqrt(np.mean(W)/k), (n, k))
    save_csv(H_init, "H_small.csv")

    # Run SymNMF
    H_final = symnmfmodule.symnmf("small_data.csv", H_csv="H_small.csv", W_csv="W_small.csv", k=k)
    print("\n--- Small dataset SymNMF H_final ---")
    print_matrix(H_final, "H_final")

    # K-means comparison
    proc = subprocess.run(
        [sys.executable, "kmeans.py", str(k)],
        input="\n".join(",".join(f"{v:.4f}" for v in row) for row in X),
        text=True,
        capture_output=True
    )
    assert proc.returncode == 0, "kmeans.py failed"
    print("\n--- Small dataset K-means centroids ---")
    print(proc.stdout.strip())

# ----------------------------
# Run all tests
# ----------------------------
if __name__ == "__main__":
    print("Testing core matrix functions via symnmfmodule...")
    test_core_functions()

    print("\nTesting SymNMF algorithm...")
    test_symnmf_algorithm()

    print("\nTesting K-means script...")
    test_kmeans()

    print("\nTesting analysis.py pipeline...")
    test_analysis_pipeline()

    print("\nTesting small synthetic dataset (stable)...")
    test_small_dataset_stable()

    print("\nAll tests passed successfully!")
