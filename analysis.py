import sys
import numpy as np
from symnmf import run
import kmeans

def run_symnmf(k, file_name):
    """Run symnmf.py with symnmf goal and return the resulting matrix as a NumPy array."""
    try:
        result = subprocess.run(
            ["python3", "./symnmf.py", str(k), "symnmf", file_name],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        matrix = [list(map(float, line.split(","))) for line in lines]
        return np.array(matrix)
    except subprocess.CalledProcessError:
        print("An Error Has Occurred")
        sys.exit(1)

def run_kmeans(k, file_name):
    """Run kmeans.py and return the resulting matrix as a NumPy array."""
    try:
        result = subprocess.run(
            ["python3", "./kmeans.py", str(k), file_name],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip().split("\n")
        matrix = [list(map(float, line.split(","))) for line in lines]
        return np.array(matrix)
    except subprocess.CalledProcessError:
        print("An Error Has Occurred")
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        file_name = sys.argv[2]
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

    # Run SymNMF
    H_nmf = run_symnmf(k, file_name)
    print("SymNMF result:")
    print(H_nmf)

    # Run K-means
    centroids = run_kmeans(k, file_name)
    print("K-means result:")
    print(centroids)

if __name__ == "__main__":
    main()