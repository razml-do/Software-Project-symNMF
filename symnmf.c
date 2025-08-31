#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_LINE_LEN 4096

#define ERROR_EXIT()                       \
    do                                     \
    {                                      \
        printf("An Error Has Occurred\n"); \
        exit(1);                           \
    } while (0)

typedef struct Matrix
{
    int rows;
    int cols;
    double **mat;
} matrix;

/**
 * create_matrix - Allocate a matrix with given rows and columns
 * @rows: number of rows
 * @cols: number of columns
 *
 * Returns a matrix with memory allocated and elements initialized to 0.
 * On allocation failure, prints "An Error Has Occurred" and exits.
 */
matrix create_matrix(int rows, int cols)
{
    int i = 0;
    matrix M;

    M.rows = rows;
    M.cols = cols;
    M.mat = NULL;
    if (M.rows == 0 || M.cols == 0)
        return M;

    M.mat = malloc(sizeof(double *) * rows);
    if (!M.mat)
        ERROR_EXIT();

    for (i = 0; i < rows; i++)
    {
        M.mat[i] = calloc(cols, sizeof(double));
        if (!M.mat[i])
            ERROR_EXIT();
    }

    return M;
}

/**
 * destroy_matrix - Free memory associated with a matrix
 * @M: pointer to matrix to destroy
 *
 * After freeing, sets mat=NULL and rows/cols=-1. Must only be used on a valid matrix.
 */
void destroy_matrix(matrix *M)
{
    int i = 0;
    for (i = 0; i < M->rows; i++)
    {
        free(M->mat[i]);
    }
    free(M->mat);

    M->mat = NULL;
    M->rows = -1;
    M->cols = -1;
}

/**
 * matrix_add_row - Add a zero-initialized row to a matrix
 * @M: pointer to matrix
 *
 * Reallocates memory for an extra row. Must only be used on a valid matrix.
 */
void copy_matrix(matrix dest, matrix src)
{
    int i, j;

    for (i = 0; i < dest.rows; i++)
    {
        for (j = 0; j < dest.cols; j++)
        {
            dest.mat[i][j] = src.mat[i][j];
        }
    }
}

/**
 * matrix_add_row - Add a zero-initialized row to a matrix
 * @M: pointer to matrix
 *
 * Reallocates memory for an extra row. Must only be used on a valid matrix.
 */
void matrix_add_row(matrix *M)
{
    M->rows++;
    M->mat = realloc(M->mat, sizeof(double *) * M->rows);
    if (!M->mat)
        ERROR_EXIT();

    M->mat[M->rows - 1] = calloc(M->cols, sizeof(double));
    if (!M->mat[M->rows - 1])
        ERROR_EXIT();
}

/**
 * print_matrix - Print a matrix to stdout with 4 decimal precision
 * @M: matrix to print
 */
void print_matrix(matrix M)
{
    int i, j;

    for (i = 0; i < M.rows; i++)
    {
        for (j = 0; j < M.cols; j++)
        {
            printf("%.4f", M.mat[i][j]);
            if (j < M.cols - 1)
            {
                printf(",");
            }
        }

        printf("\n");
    }
}

/**
 * squared_euclidean_distance - Compute squared Euclidean distance between two vectors
 * @p: first vector
 * @q: second vector
 * @d: dimension of vectors
 *
 * Returns sum((p_i - q_i)^2) for i=0..d-1
 */
double squared_euclidean_distance(double *p, double *q, int d)
{
    int i;
    double squareSum = 0;

    for (i = 0; i < d; i++)
        squareSum += (p[i] - q[i]) * (p[i] - q[i]);

    return squareSum;
}

/**
 * mat_const_mul_and_add - Compute dest = mul*M + add element-wise
 * @dest: output matrix
 * @M: input matrix
 * @mul: multiplicative constant
 * @add: additive constant
 */
void mat_const_mul_and_add(matrix dest, matrix M, double mul, double add)
{
    int row, col;
    for (row = 0; row < M.rows; row++)
    {
        for (col = 0; col < M.cols; col++)
        {
            dest.mat[row][col] = (mul * M.mat[row][col]) + add;
        }
    }
}

/**
 * mat_transpose - Compute transpose of M into dest
 * @dest: output matrix
 * @M: input matrix
 */
void mat_transpose(matrix dest, matrix M)
{
    int row, col;

    for (row = 0; row < M.rows; row++)
    {
        for (col = 0; col < M.cols; col++)
        {
            dest.mat[col][row] = M.mat[row][col];
        }
    }
}

/**
 * mat_sub - Compute element-wise subtraction: dest = M1 - M2
 * @dest: output matrix
 * @M1: first input matrix
 * @M2: second input matrix
 */
void mat_sub(matrix dest, matrix M1, matrix M2)
{
    int row, col;

    for (row = 0; row < M1.rows; row++)
    {
        for (col = 0; col < M1.cols; col++)
        {
            dest.mat[row][col] = M1.mat[row][col] - M2.mat[row][col];
        }
    }
}

/**
 * mat_mul_indices - Compute element-wise multiplication: dest = M1 * M2
 * @dest: output matrix
 * @M1: first input matrix
 * @M2: second input matrix
 *
 * Exits on any zero in M2.
 */
void mat_mul_indices(matrix dest, matrix M1, matrix M2)
{
    int row, col;

    for (row = 0; row < M1.rows; row++)
    {
        for (col = 0; col < M1.cols; col++)
        {
            if (M2.mat[row][col] == 0)
                ERROR_EXIT();

            dest.mat[row][col] = M1.mat[row][col] * M2.mat[row][col];
        }
    }
}

/**
 * mat_div - Compute element-wise division: dest = M1 / M2
 * @dest: output matrix
 * @M1: numerator matrix
 * @M2: denominator matrix
 *
 * Exits if any element of M2 is zero.
 */
void mat_div(matrix dest, matrix M1, matrix M2)
{
    int row, col;

    for (row = 0; row < M1.rows; row++)
    {
        for (col = 0; col < M1.cols; col++)
        {
            if (M2.mat[row][col] == 0)
                ERROR_EXIT();

            dest.mat[row][col] = M1.mat[row][col] / M2.mat[row][col];
        }
    }
}

/**
 * frobenius_norm_squared - Compute squared Frobenius norm of a matrix
 * @M: input matrix
 *
 * Returns sum of squares of all elements.
 */
double frobenius_norm_squared(matrix M)
{
    int row, col;
    double squared_forbenius_norm = 0;
    for (row = 0; row < M.rows; row++)
    {
        for (col = 0; col < M.cols; col++)
        {
            squared_forbenius_norm += M.mat[row][col] * M.mat[row][col];
        }
    }
    return squared_forbenius_norm;
}

/**
 * vertex_degree - Compute sum of elements in a vector
 * @row: input row vector
 * @d: dimension
 *
 * Returns the sum of the vector elements.
 */
double vertex_degree(double *row, int d)
{
    double deg = 0;
    int i;

    for (i = 0; i < d; i++)
        deg += row[i];

    return deg;
}

/**
 * diagonal_degree_matrix - Compute diagonal degree matrix from similarity matrix
 * @dest: output matrix (diagonal)
 * @sim: input similarity matrix
 */
void diagonal_degree_matrix(matrix dest, matrix sim)
{
    int i;

    /*mat is diagonal, only operate on elements of the diagonal.*/
    for (i = 0; i < sim.rows; i++)
    {
        dest.mat[i][i] = vertex_degree(sim.mat[i], sim.cols);
    }
}

/**
 * similarity_matrix - Compute Gaussian similarity (RBF kernel) matrix
 * @dest: output matrix
 * @X: input data matrix
 *
 * dest(i,j) = exp(-||X_i - X_j||^2 / 2) for i != j, 0 on diagonal
 */
void similarity_matrix(matrix dest, matrix X)
{
    int row, col;
    for (row = 0; row < X.rows; row++)
    {
        for (col = 0; col < X.rows; col++)
        {
            if (row != col)
                dest.mat[row][col] = exp(-squared_euclidean_distance(X.mat[row], X.mat[col], X.cols) / 2.0);
        }
    }
}

/*Only diagonal elements are affected as each becomes D_ii = 1/sqrt(D_ii)*/
void diagonal_inverse_square_root(matrix dest, matrix D)
{
    int i;

    for (i = 0; i < D.rows; i++)
    {
        dest.mat[i][i] = 1 / sqrt(D.mat[i][i]);
    }
}

/**
 * mat_mul - Standard matrix multiplication: dest = M1 * M2
 * @dest: output matrix
 * @M1: left matrix
 * @M2: right matrix
 *
 * Exits if M1.cols != M2.rows
 */
void mat_mul(matrix dest, matrix M1, matrix M2)
{
    int i, j, k;

    if (M1.cols != M2.rows)
        ERROR_EXIT(); /* invalid dimensions */

    for (i = 0; i < M1.rows; i++)
    {
        for (j = 0; j < M2.cols; j++)
        {
            dest.mat[i][j] = 0;
            for (k = 0; k < M1.cols; k++)
            {
                dest.mat[i][j] += M1.mat[i][k] * M2.mat[k][j];
            }
        }
    }
}

/**
 * count_commas - Count number of commas in a string
 * @line: input string
 *
 * Returns number of commas found.
 */
int count_commas(const char *line)
{
    int i = 0;
    int count = 0;
    for (i = 0; line[i] != '\0'; i++)
    {
        if (line[i] == ',')
            count++;
    }
    return count;
}

/**
 * read_mat_from_csv_file - Read CSV file into a matrix
 * @filename: path to CSV file
 * @M: matrix to fill (should be 0x0 initially)
 *
 * Allocates rows dynamically. Exits on any error.
 */
void read_mat_from_csv_file(const char *filename, matrix *M)
{
    char line[MAX_LINE_LEN];
    int i;
    char *token;
    FILE *file = fopen(filename, "r");

    if (!file)
        ERROR_EXIT();

    while (fgets(line, sizeof(line), file))
    {
        if (M->cols == 0)
            M->cols = count_commas(line) + 1;

        matrix_add_row(M);

        token = strtok(line, ",");
        for (i = 0; i < M->cols; i++)
        {
            if (!token)
            {
                ERROR_EXIT();
            }

            M->mat[M->rows - 1][i] = atof(token);
            token = strtok(NULL, ",");
        }
    }

    fclose(file);
}

/**
 * average_of_entries - Compute average of all elements in a matrix
 * @W: input matrix
 */
double average_of_entries(matrix W)
{
    int i, j;
    double avg = 0.0;
    for (i = 0; i < W.rows; i++)
        for (j = 0; j < W.cols; j++)
            avg += W.mat[i][j];
    avg /= (W.rows * W.cols);
    return avg;
}

/**
 * H_initilization - Initialize H matrix with random values
 * @H: matrix to initialize
 * @m: scaling factor (average entry in similarity matrix)
 * @k: number of clusters
 *
 * Random values in [0, 2*sqrt(m/k)]
 */
void H_initilization(matrix H, double m, int k)
{
    int row, col;
    for (row = 0; row < H.rows; row++)
    {
        for (col = 0; col < H.cols; col++)
        {
            H.mat[row][col] = ((double)rand() / RAND_MAX) * 2 * sqrt(m / k);
        }
    }
}

/**
 * sim_strategy - Compute similarity matrix from X
 * @dest: output matrix
 * @X: input matrix
 */
void sim_strategy(matrix dest, matrix X)
{
    similarity_matrix(dest, X);
}

/**
 * ddg_strategy - Compute diagonal degree matrix from X
 * @dest: output matrix
 * @X: input matrix
 */
void ddg_strategy(matrix dest, matrix X)
{
    matrix A = create_matrix(X.rows, X.rows);

    similarity_matrix(A, X);
    diagonal_degree_matrix(dest, A);

    destroy_matrix(&A);
}

/**
 * norm_strategy - Compute normalized similarity matrix: W = D^{-1/2} * A * D^{-1/2}
 * @dest: output matrix
 * @X: input matrix
 */
void norm_strategy(matrix dest, matrix X)
{
    matrix A = create_matrix(X.rows, X.rows);
    matrix D = create_matrix(X.rows, X.rows);
    matrix invsqrD = create_matrix(X.rows, X.rows);
    matrix invsqrD_A = create_matrix(X.rows, X.rows);

    similarity_matrix(A, X);
    diagonal_degree_matrix(D, A);

    diagonal_inverse_square_root(invsqrD, D);
    mat_mul(invsqrD_A, invsqrD, A);
    /*W = normalized similarity matrix (W = invsqrD*A*invsqrD) */
    mat_mul(dest, invsqrD_A, invsqrD);

    destroy_matrix(&A);
    destroy_matrix(&D);
    destroy_matrix(&invsqrD);
    destroy_matrix(&invsqrD_A);
}

/**
 * perform_update_step - Perform one iteration of SymNMF multiplicative update
 * @H: [in/out] current H matrix (updated in-place at the end)
 * @W: normalized similarity matrix
 * @k: number of clusters
 * @beta: mixing parameter
 * @buffers: array of temporary matrices, already allocated
 *
 * Returns: squared Frobenius norm of (H_new - H).
 */
static double perform_update_step(matrix H, matrix W, double beta, matrix buffers[8])
{
    matrix WH = buffers[0], H_transpose = buffers[1];
    matrix HHT = buffers[2], HHTH = buffers[3];
    matrix ratio_mat = buffers[4], beta_mul_and_addition = buffers[5];
    matrix frob_sub = buffers[6], H_new = buffers[7];

    mat_mul(WH, W, H);
    mat_transpose(H_transpose, H);
    mat_mul(HHT, H, H_transpose);
    mat_mul(HHTH, HHT, H);
    mat_div(ratio_mat, WH, HHTH);
    mat_const_mul_and_add(beta_mul_and_addition, ratio_mat, beta, 1 - beta);
    mat_mul_indices(H_new, H, beta_mul_and_addition);
    mat_sub(frob_sub, H_new, H);

    copy_matrix(H, H_new);
    return frobenius_norm_squared(frob_sub);
}

/**
 * symnmf_strategy - Perform symnmf optimization
 * @final_H: [out] resulting H matrix
 * @k: number of clusters
 * @W: normalized similarity matrix
 * @H: initialized H matrix
 *
 * Runs multiplicative update rules until convergence or max_iter.
 */
void symnmf_strategy(matrix final_H, int k, matrix W, matrix H)
{
    const int max_iter = 300;
    const double beta = 0.5, epsilon = 1e-4;
    int i;
    double squared_frob;

    /* Preallocate all temporary buffers once */
    matrix buffers[8];
    buffers[0] = create_matrix(W.rows, k);      /* WH */
    buffers[1] = create_matrix(k, W.rows);      /* H_transpose */
    buffers[2] = create_matrix(W.rows, W.rows); /* HHT */
    buffers[3] = create_matrix(W.rows, k);      /* HHTH */
    buffers[4] = create_matrix(W.rows, k);      /* ratio_mat */
    buffers[5] = create_matrix(W.rows, k);      /* beta_mul_and_addition */
    buffers[6] = create_matrix(W.rows, k);      /* frob_sub */
    buffers[7] = create_matrix(W.rows, k);      /* H_new */

    for (i = 0; i < max_iter; i++)
    {
        squared_frob = perform_update_step(H, W, beta, buffers);
        if (squared_frob < epsilon)
            break;
    }
    copy_matrix(final_H, H);
    for (i = 0; i < 8; i++)
        destroy_matrix(&buffers[i]);
}

/**
 * initialize_for_symnmf - Prepare matrices for SymNMF optimization
 * @X: input data matrix
 * @k: number of clusters
 * @W: [out] allocated normalized similarity matrix
 * @H: [out] allocated and initialized H matrix
 * @result: [out] allocated matrix for final H
 *
 * Allocates and initializes all matrices required for SymNMF.
 * The caller is responsible for destroying these matrices after use.
 */
void initialize_for_symnmf(matrix X, int k, matrix *W, matrix *H, matrix *result)
{
    double m;

    /* Allocate and compute normalized similarity matrix */
    *W = create_matrix(X.rows, X.rows);
    norm_strategy(*W, X);

    /* Compute average entry in W */
    m = average_of_entries(*W);

    /* Allocate and initialize H with random values in [0, 2*sqrt(m/k)] */
    *H = create_matrix(X.rows, k);
    H_initilization(*H, m, k);

    /* Allocate matrix to store final H */
    *result = create_matrix(X.rows, k);
}

/**
 * get_mat_of_strategy - Compute a matrix according to the chosen strategy
 * @X: input data matrix
 * @strategy: one of "sym", "ddg", "norm", or "symnmf"
 * @k: number of clusters (only used for "symnmf")
 *
 * Returns a newly allocated matrix corresponding to the chosen strategy:
 *   - "sym": similarity matrix
 *   - "ddg": diagonal degree matrix
 *   - "norm": normalized similarity matrix
 *   - "symnmf": optimized H matrix using SymNMF
 *
 * The caller is responsible for destroying the returned matrix.
 */
matrix get_mat_of_strategy(matrix X, char *strategy, int k)
{
    /* Optional parameters for symnmf. */
    matrix W, H, result;

    /* Symnmf optimization process */
    if (strcmp(strategy, "symnmf") == 0)
    {
        initialize_for_symnmf(X, k, &W, &H, &result); /* Calculate final H using the symnmf algorithm */
        symnmf_strategy(result, k, W, H);

        destroy_matrix(&W);
        destroy_matrix(&H);
    }
    else
    {
        result = create_matrix(X.rows, X.rows);
        if ((strcmp(strategy, "sym") == 0))
            sim_strategy(result, X);
        else if ((strcmp(strategy, "ddg") == 0))
            ddg_strategy(result, X);
        else if ((strcmp(strategy, "norm") == 0))
            norm_strategy(result, X);
    }
    return result;
}

/**
 * parse_arguments - Validate and extract parameters from argv
 * @argc: argument count
 * @argv: argument vector
 * @k: [out] number of clusters (only for symnmf)
 * @filename: [out] pointer to filename string
 * @strategy: [out] pointer to chosen strategy string
 *
 * Exits with "An Error Has Occurred" on invalid input.
 */
void parse_arguments(int argc, char *argv[], int *k, char **filename, char **strategy)
{
    int sym, ddg, norm, symnmf;

    if (argc != 3 && argc != 4)
        ERROR_EXIT();

    *strategy = argv[1];
    sym = (strcmp(argv[1], "sym") == 0);
    ddg = (strcmp(argv[1], "ddg") == 0);
    norm = (strcmp(argv[1], "norm") == 0);
    symnmf = (strcmp(argv[1], "symnmf") == 0);

    if (sym || ddg || norm)
    {
        if (argc != 3)
            ERROR_EXIT();
        *filename = argv[2];
        *k = 1; /* not used */
    }
    else if (symnmf)
    {
        if (argc != 4)
            ERROR_EXIT();
        *k = atoi(argv[2]);
        *filename = argv[3];
    }
    else
    {
        ERROR_EXIT(); /* invalid strategy */
    }
}

int main(int argc, char *argv[])
{
    int k = 1; /* required number of clusters */
    char *filename = NULL;
    char *strategy = argv[1]; /* sym, ddg, norm or symnmf */
    matrix X, mat_to_print;
    /* Parse command-line arguments */
    parse_arguments(argc, argv, &k, &filename, &strategy);
    strategy = argv[1];

    /* Read input CSV file */
    X = create_matrix(0, 0);
    read_mat_from_csv_file(filename, &X);

    /* Calculate desired matrix */
    mat_to_print = get_mat_of_strategy(X, strategy, k);
    print_matrix(mat_to_print);
    destroy_matrix(&X);
    destroy_matrix(&mat_to_print);
    return 0;
}
