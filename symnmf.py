import sys
import numpy as np
import symnmfmodule

def parse_input(file_name):
    """
    Read the input file into a NumPy array.

    :param file_name: input data
    """
    X = np.loadtxt(file_name, delimiter=',')
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def print_matrix(matrix):
    """
    Print matrix with required formatting.

    :param matrix: input matrix to print
    """
    for row in matrix:
        print(",".join([f"{val:.4f}" for val in row]))

def symnmf(X: list[list[float]], k: int):
    """
    symnmf algorithm using symnmfmodule.c.

    :param X: input data
    :param k: required number of clusters
    """
    if k is None:
            print("An Error Has Occurred")
            sys.exit(1)
    # Calculate the normalized similarity matrix W.
    if isinstance(X, np.ndarray):
        X = X.tolist()
    W = symnmfmodule.norm(X)
    # mean of all entries in normalized similarity matrix
    m = np.mean(W)
    
    np.random.seed(1234) # The random seed required by the project.
    H = np.random.uniform(0, 2.0 * np.sqrt(m / k), (len(X), k)).tolist()
    H_final = symnmfmodule.symnmf(k, W, H)     
    return H_final   

def main():
    """
    Main algorithms using symnmfmodule.c.
    cases for goal:
        sym -> print similarity matrix
        ddg -> print diagonal degree matrix
        norm -> print normalized similarity matrix
        symnmf -> print symnmf matrix
    """
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
            
        print_matrix(symnmf(X, k))

if __name__ == "__main__":
    main()