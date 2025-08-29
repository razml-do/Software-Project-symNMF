import sys
import numpy as np
import symnmfmodule

symnmfmodule.sim() 


def parse_input(file_name):
    """Read the input file into a NumPy array"""
    return np.loadtxt(file_name, delimiter=',')

def print_matrix(matrix):
    """Print matrix with required formatting"""
    for row in matrix:
        print(",".join([f"{val:.4f}" for val in row]))

def analysis_symnmf(X, k):
    """Run the exact symnmf algorithm in main and get H_final (for analysis.py)"""
    if k is None:
            print("An Error Has Occurred")
            sys.exit(1)
            
    # Calculate the normalized similarity matrix W.
    W = symnmfmodule.norm(X.tolist())
    # mean of all entries in normalized similarity matrix
    m = np.mean(W)
    
    # Calculate the initial matrix H.
    np.random.seed(1234) # The random seed required by the project.
    H = np.random.uniform(0, 2.0 * np.sqrt(m / k), (X.shape[0], k))
    
    H_final = symnmfmodule.symnmf(k, W, H.tolist())

    return H_final


def main():
    args = sys.argv[1:]

    if len(args) == 2:
        # Case: goal + file_name (no k needed)
        goal = args[0]
        file_name = args[1]
        k = None
    elif len(args) == 3:
        # Case: k + goal + file_name
        try:
            k = int(args[0])
        except ValueError:
            print("An Error Has Occurred")
            sys.exit(1)
        goal = args[1]
        file_name = args[2]
    else:
        print("An Error Has Occurred")
        sys.exit(1)

    X = parse_input(file_name)

    if goal == "sym":
        A = symnmfmodule.sym(X.tolist())
        print_matrix(A)

    elif goal == "ddg":
        D = symnmfmodule.ddg(X.tolist())
        print_matrix(D)

    elif goal == "norm":
        W = symnmfmodule.norm(X.tolist())
        print_matrix(W)

    elif goal == "symnmf":
        if k is None:
            print("An Error Has Occurred")
            sys.exit(1)
            
        # Calculate the normalized similarity matrix W.
        W = symnmfmodule.norm(X.tolist())
        # mean of all entries in normalized similarity matrix
        m = np.mean(W)
        
        # Calculate the initial matrix H.
        np.random.seed(1234) # The random seed required by the project.
        H = np.random.uniform(0, 2.0 * np.sqrt(m / k), (X.shape[0], k))
        
        H_final = symnmfmodule.symnmf(k, W, H.tolist())
        print_matrix(H_final)

    else:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()
