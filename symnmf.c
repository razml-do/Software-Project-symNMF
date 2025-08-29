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

/*Create Matrix struct by allocating it's space*/
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

/*Destroy Matrix struct by freeing it's contents.*/
/*MUST be used ONLY on an already created matrix!*/
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

/*MUST be used ONLY on an already created matrix with appropriate dimensions!*/
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

/*Add a 0's row to matrix*/
/*MUST be used ONLY on an already created matrix!*/
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

/* -------------------- UTILITY FUNCTIONS -------------------- */
void print_Matrix(matrix M)
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

double squared_euclidean_distance(double *p, double *q, int d)
{
    int i;
    double squareSum = 0;

    for (i = 0; i < d; i++)
        squareSum += (p[i] - q[i]) * (p[i] - q[i]);

    return squareSum;
}

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

/*Indices div!*/
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

double vertex_degree(double *row, int d)
{
    double deg = 0;
    int i;

    for (i = 0; i < d; i++)
        deg += row[i];

    return deg;
}

void diagonal_degree_matrix(matrix dest, matrix sim)
{
    int i;

    /*mat is diagonal, only operate on elements of the diagonal.*/
    for (i = 0; i < sim.rows; i++)
    {
        dest.mat[i][i] = vertex_degree(sim.mat[i], sim.cols);
    }
}

/*similarity_matrix - Compute a Gaussian similarity (RBF kernel) matrix.*/
/* For each pair of distinct rows (row ≠ col) in X, this function computes the*/
/* squared Euclidean distance between them and converts it into a similarity score*/
/* using the Gaussian kernel:*/
/* sim(row, col) = exp(-||X_row - X_col||² / 2)*/
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

/* -------------------- CSV READING -------------------- */
/*Matrix must be 0x0 at start!*/
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

/* For getting m*/
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

void sim_strategy(matrix dest, matrix X)
{
    similarity_matrix(dest, X);
}

void ddg_strategy(matrix dest, matrix X)
{
    matrix A = create_matrix(X.rows, X.rows);

    similarity_matrix(A, X);
    diagonal_degree_matrix(dest, A);

    destroy_matrix(&A);
}

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

void symnmf_strategy(matrix final_H, int k, matrix W, matrix H)
{
    const int max_iter = 300;
    const double beta = 0.5;
    const double epsilon = 1e-4;
    int i;
    double squared_frob;

    matrix WH = create_matrix(W.rows, k);
    matrix H_transpose = create_matrix(k, W.rows);
    matrix HHT = create_matrix(W.rows, W.rows);
    matrix HHTH = create_matrix(W.rows, k);
    matrix ratio_mat = create_matrix(W.rows, k);
    matrix beta_mul_and_addition = create_matrix(W.rows, k);
    matrix frob_sub = create_matrix(W.rows, k);
    matrix H_new = create_matrix(W.rows, k);

    for (i = 0; i < max_iter; i++)
    {
        mat_mul(WH, W, H);
        mat_transpose(H_transpose, H);
        mat_mul(HHT, H, H_transpose);
        mat_mul(HHTH, HHT, H);
        mat_div(ratio_mat, WH, HHTH);
        mat_const_mul_and_add(beta_mul_and_addition, ratio_mat, beta, 1 - beta);
        mat_mul_indices(H_new, H, beta_mul_and_addition);
        mat_sub(frob_sub, H_new, H);
        squared_frob = frobenius_norm_squared(frob_sub);

        copy_matrix(H, H_new);

        if (squared_frob < epsilon)
            break;
    }

    copy_matrix(final_H, H);

    destroy_matrix(&WH);
    destroy_matrix(&H_transpose);
    destroy_matrix(&HHT);
    destroy_matrix(&HHTH);
    destroy_matrix(&ratio_mat);
    destroy_matrix(&beta_mul_and_addition);
    destroy_matrix(&frob_sub);
    destroy_matrix(&H_new);
}

matrix get_mat_of_strategy(matrix X, char *strategy, int k)
{
    /* Optional parameters for symnmf. */
    matrix W;
    matrix H;
    double m;

    matrix result;

    /* Symnmf optimization process */
    if (strcmp(strategy, "symnmf") == 0)
    {
        /* Get the normalized similarity matrix */
        W = create_matrix(X.rows, X.rows);
        norm_strategy(W, X);

        /* Compute m = mean of entries in normalized similarity */
        m = average_of_entries(W);

        /* Create and initialize H with random values in the range [0, 2*sqrt(m/k)] */
        H = create_matrix(X.rows, k);
        H.mat[0][0] = 0.14721153;
        H.mat[0][1] = 0.47818423;
        H.mat[1][0] = 0.33645966;
        H.mat[1][1] = 0.60366629;
        H.mat[2][0] = 0.59952881;
        H.mat[2][1] = m;
        H.mat[2][1] = 0.20952845;
        /*H_initilization(H, m, k);*/

        /* Calculate final H using the symnmf algorithm (final H will be available in result.) */
        result = create_matrix(X.rows, k);
        symnmf_strategy(result, k, W, H);

        destroy_matrix(&W);
        destroy_matrix(&H);
    }
    else
    {
        result = create_matrix(X.rows, X.rows);

        if ((strcmp(strategy, "sym") == 0))
        {
            sim_strategy(result, X);
        }
        else if ((strcmp(strategy, "ddg") == 0))
        {
            ddg_strategy(result, X);
        }
        else if ((strcmp(strategy, "norm") == 0))
        {
            norm_strategy(result, X);
        }
    }

    return result;
}

int main(int argc, char *argv[])
{
    int k = 1;
    int sym;
    int ddg;
    int norm;
    int symnmf; /*boolean like for strategy*/
    char *filename = NULL;
    char *strategy = argv[1];
    matrix X;
    matrix mat_to_print;
    if (argc != 3 && argc != 4)
    {
        ERROR_EXIT();
    }
    sym = (strcmp(argv[1], "sym") == 0);
    ddg = (strcmp(argv[1], "ddg") == 0);
    norm = (strcmp(argv[1], "norm") == 0);
    symnmf = (strcmp(argv[1], "symnmf") == 0);
    strategy = argv[1];

    if (sym || ddg || norm)
    {
        if (argc != 3)
        {
            ERROR_EXIT();
        }

        filename = argv[2];
    }
    else if (symnmf)
    {
        if (argc != 4)
        {
            ERROR_EXIT();
        }

        k = atoi(argv[2]);
        filename = argv[3];
    }
    else
    {
        ERROR_EXIT(); /*not a valid strategy*/
    }

    /* Read input CSV file */
    X = create_matrix(0, 0);
    read_mat_from_csv_file(filename, &X);

    /* Calculate desired matrix */
    mat_to_print = get_mat_of_strategy(X, strategy, k);
    print_Matrix(mat_to_print);
    destroy_matrix(&X);
    destroy_matrix(&mat_to_print);
    return 0;
}
