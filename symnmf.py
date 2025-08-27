import sys
import numpy as np
import symnmf  

np.random.seed(1234)

def parse_input(file_name):
    """Read the input file into a NumPy array"""
    return np.loadtxt(file_name, delimiter=',')

def print_matrix(matrix):
    """Print matrix with required formatting"""
    for row in matrix:
        print(",".join([f"{val:.4f}" for val in row]))

def main():
    if len(sys.argv) != 4:
        print("Invalid Input!")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        file_name = sys.argv[3]
    except Exception:
        print("Invalid Input!")
        sys.exit(1)

    X = parse_input(file_name)

    if goal == "sym":
        W = symnmf.sym(X)
        print_matrix(W)

    elif goal == "ddg":
        D = symnmf.ddg(X)
        print_matrix(D)

    elif goal == "norm":
        normW = symnmf.norm(X)
        print_matrix(normW)

    elif goal == "symnmf":
        # Build similarity matrix and initialize H
        W = symnmf.sym(X)
        m = np.mean(W)  # mean of all entries in normalized similarity matrix
        H = np.random.uniform(0, 2.0 * np.sqrt(m / k), (X.shape[0], k))

        # Save temporary CSV files to match C wrapper
        np.savetxt("H_init.csv", H, delimiter=",")
        np.savetxt("W.csv", W, delimiter=",")

        # Call C SymNMF with filenames
        H_final = symnmf.symnmf(file_name, "H_init.csv", "W.csv", k)


        # Optional: fix empty clusters
        if np.any(H_final.sum(axis=0) < 1e-6):
            H_final += 1e-4

        print_matrix(H_final)

    else:
        print("Invalid Input!")
        sys.exit(1)

if __name__ == "__main__":
    main()
