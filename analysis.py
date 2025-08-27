import sys
import numpy as np
import subprocess
import symnmfmodule  # type: ignore

def safe_silhouette_score(data, labels):
    """Return silhouette score, or nan if only 1 cluster exists."""
    from sklearn.metrics import silhouette_score
    if len(np.unique(labels)) < 2:
        return 0.0
    return silhouette_score(data, labels)

def read_txt_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append([float(x) for x in line.split(",")])
    return np.array(data)

def run_kmeans(data, k):
    # Run your existing kmeans.py as a subprocess to get centroids
    proc = subprocess.run(
        [sys.executable, "kmeans.py", str(k)],
        input="\n".join(",".join(map(str, row)) for row in data),
        text=True,
        capture_output=True
    )

    if proc.returncode != 0:
        print("An Error Has Occurred in K-means")
        sys.exit(1)

    # Parse centroids output
    centroids = []
    for line in proc.stdout.strip().splitlines():
        centroids.append([float(x) for x in line.strip().split(",")])
    centroids = np.array(centroids)

    # Assign each point to nearest centroid
    clusters = []
    for x in data:
        dists = np.linalg.norm(centroids - x, axis=1)
        clusters.append(np.argmin(dists))
    return np.array(clusters)

def run_symnmf(data, k):
    # Save input data to CSV
    np.savetxt("data.csv", data, delimiter=",")

    # Build similarity matrix
    W = symnmfmodule.sym("data.csv")
    np.savetxt("W.csv", W, delimiter=",")

    # Initialize H
    M = np.mean(W)
    H_init = np.random.uniform(0, 2 * np.sqrt(M / k), (data.shape[0], k))
    np.savetxt("H.csv", H_init, delimiter=",")

    # Call C SymNMF wrapper
    H_final = symnmfmodule.symnmf("data.csv", H_csv="H.csv", W_csv="W.csv", k=k)
    clusters = np.argmax(H_final, axis=1)
    return clusters

def main():
    k = int(sys.argv[1])
    file_name = sys.argv[2]

    data = read_txt_file(file_name)

    # Run SymNMF
    nmf_labels = run_symnmf(data, k)
    nmf_score = safe_silhouette_score(data, nmf_labels)

    # Run K-means
    kmeans_labels = run_kmeans(data, k)
    kmeans_score = safe_silhouette_score(data, kmeans_labels)

    print(f"nmf: {nmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")

if __name__ == "__main__":
    main()
