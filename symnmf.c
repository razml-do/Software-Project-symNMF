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

typedef struct
{
    int rows;
    int cols;
    double **mat;
} *Matrix;

/*Create Matrix struct by allocating it's space*/
Matrix create_matrix(int rows, int cols)
{
    int i = 0;
    Matrix M;
    M = malloc(sizeof(*M));
    if (!M)
        ERROR_EXIT();
    M->rows = rows;
    M->cols = cols;
    M->mat = NULL;
    if (M->rows == 0 || M->cols == 0)
        return M;
    M->mat = malloc(sizeof(double *) * rows);
    if (!M->mat)
        ERROR_EXIT();
    for (i = 0; i < rows; i++)
    {
        M->mat[i] = calloc(cols, sizeof(double));
        if (!M->mat[i])
            ERROR_EXIT();
    }
    return M;
}

/*Destroy Matrix struct by freeing it's contents.*/
/*MUST be used ONLY on an already created matrix!*/
void destroy_matrix(Matrix M)
{
    int i = 0;
    for (i = 0; i < M->rows; i++)
    {
        free(M->mat[i]);
    }
    free(M->mat);
    free(M);
}

/*MUST be used ONLY on an already created matrix with appropriate dimensions!*/
void copy_matrix(Matrix dest, Matrix src)
{
    int i, j;

    for (i = 0; i < dest->rows; i++)
    {
        for (j = 0; j < dest->cols; j++)
        {
            dest->mat[i][j] = src->mat[i][j];
        }
    }
}

/*Add a 0's row to matrix*/
/*MUST be used ONLY on an already created matrix!*/
void add_row(Matrix M)
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
void print_Matrix(Matrix M)
{
    int i, j;

    for (i = 0; i < M->rows; i++)
    {
        for (j = 0; j < M->cols; j++)
        {
            printf("%.4f", M->mat[i][j]);
            if (j < M->cols - 1)
                printf(",");
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

void mat_const_mul_and_add(Matrix dest, Matrix M, double mul, double add)
{
    int row, col;
    for (row = 0; row < M->rows; row++)
    {
        for (col = 0; col < M->cols; col++)
        {
            dest->mat[row][col] = (mul * M->mat[row][col]) + add;
        }
    }
}

void mat_transpose(Matrix dest, Matrix M)
{
    int row, col;

    for (row = 0; row < M->rows; row++)
    {
        for (col = 0; col < M->cols; col++)
            dest->mat[col][row] = M->mat[row][col];
    }
}

void mat_sub(Matrix dest, Matrix M1, Matrix M2)
{
    int row, col;

    for (row = 0; row < M1->rows; row++)
    {
        for (col = 0; col < M1->cols; col++)
        {
            dest->mat[row][col] = M1->mat[row][col] - M2->mat[row][col];
        }
    }
}

void mat_mul_indices(Matrix dest, Matrix M1, Matrix M2)
{
    int row, col;

    for (row = 0; row < M1->rows; row++)
    {
        for (col = 0; col < M1->cols; col++)
        {
            if (M2->mat[row][col] == 0)
                ERROR_EXIT();

            dest->mat[row][col] = M1->mat[row][col] * M2->mat[row][col];
        }
    }
}

/*Indices div!*/
void mat_div(Matrix dest, Matrix M1, Matrix M2)
{
    int row, col;

    for (row = 0; row < M1->rows; row++)
    {
        for (col = 0; col < M1->cols; col++)
        {
            if (M2->mat[row][col] == 0)
                ERROR_EXIT();

            dest->mat[row][col] = M1->mat[row][col] / M2->mat[row][col];
        }
    }
}

double frobenius_norm_squared(Matrix M)
{
    int row, col;
    double squared_forbenius_norm = 0;
    for (row = 0; row < M->rows; row++)
    {
        for (col = 0; col < M->cols; col++)
        {
            squared_forbenius_norm += M->mat[row][col] * M->mat[row][col];
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

void diagonal_degree_matrix(Matrix dest, Matrix sim)
{
    int i;

    /*mat is diagonal, only operate on elements of the diagonal.*/
    for (i = 0; i < sim->rows; i++)
    {
        dest->mat[i][i] = vertex_degree(sim->mat[i], sim->cols);
    }
}

/*similarity_matrix - Compute a Gaussian similarity (RBF kernel) matrix.*/
/* For each pair of distinct rows (row ≠ col) in X, this function computes the*/
/* squared Euclidean distance between them and converts it into a similarity score*/
/* using the Gaussian kernel:*/
/* sim(row, col) = exp(-||X_row - X_col||² / 2)*/
void similarity_matrix(Matrix dest, Matrix X)
{
    int row, col;
    for (row = 0; row < X->rows; row++)
    {
        for (col = 0; col < X->cols; col++)
        {
            if (row != col)
                dest->mat[row][col] = exp(-squared_euclidean_distance(X->mat[row], X->mat[col], X->cols) / 2.0);
        }
    }
}

/*Only diagonal elements are affected as each becomes D_ii = 1/sqrt(D_ii)*/
void diagonal_inverse_square_root(Matrix dest, Matrix D)
{
    int i;

    for (i = 0; i < D->rows; i++)
    {
        dest->mat[i][i] = 1 / sqrt(D->mat[i][i]);
    }
}

void mat_mul(Matrix dest, Matrix M1, Matrix M2)
{
    int i, j, k;

    if (M1->cols != M2->rows)
        ERROR_EXIT(); /* invalid dimensions */

    for (i = 0; i < M1->rows; i++)
    {
        for (j = 0; j < M2->cols; j++)
        {
            dest->mat[i][j] = 0;
            for (k = 0; k < M1->cols; k++)
            {
                dest->mat[i][j] += M1->mat[i][k] * M2->mat[k][j];
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
void read_mat_from_csv_file(const char *filename, Matrix M)
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
        add_row(M);
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
double average_of_entries(Matrix W)
{
    int i, j;
    double avg = 0.0;
    for (i = 0; i < W->rows; i++)
        for (j = 0; j < W->cols; j++)
            avg += W->mat[i][j];
    avg /= (W->rows * W->cols);
    return avg;
}

void H_initilization(Matrix H, double m, int k)
{
    int row, col;
    for (row = 0; row < H->rows; row++)
    {
        for (col = 0; col < H->cols; col++)
        {
            H->mat[row][col] = ((double)rand() / RAND_MAX) * 2 * sqrt(m / k);
        }
    }
}

void get_mat_of_strategy(Matrix dest, Matrix X, char *strategy, int k)
{
    int i;
    const int max_iter = 300;
    const double beta = 0.5, epsilon = 1e-4;
    double squared_frob, m;
    /*TODO: make sure matrix dims are valid!*/
    /*k is 0 when startegy is not symnmf*/
    /*so the creation of relevant matrices is ok*/
    /*(checked inside create_matrix())*/
    Matrix A = create_matrix(X->rows, X->rows), D = create_matrix(X->rows, X->rows), W = create_matrix(X->rows, X->rows), invsqrD_Mul_A = create_matrix(X->rows, X->rows), invsqrD = create_matrix(X->rows, X->rows);
    Matrix H = create_matrix(X->rows, k), WH = create_matrix(X->rows, k), H_transpose = create_matrix(k, X->rows), HHT = create_matrix(X->rows, X->rows), HHTH = create_matrix(X->rows, k), ratio_mat = create_matrix(X->rows, k);
    Matrix beta_mul_and_addition = create_matrix(X->rows, k), frob_sub = create_matrix(X->rows, k), H_new = create_matrix(X->rows, k), invsqrD_A = create_matrix(X->rows, X->rows);

    /*A = similarity matrix*/
    similarity_matrix(A, X);
    /*D = diagonal degree matrix*/
    diagonal_degree_matrix(D, A);

    diagonal_inverse_square_root(invsqrD, D);
    mat_mul(invsqrD_A, invsqrD, A);
    /*W = normalized similarity matrix (W = invsqrD*A*invsqrD) */
    mat_mul(W, invsqrD_A, invsqrD);

    /* Symnmf optimization process */
    if (strcmp(strategy, "symnmf") == 0)
    {
        /* Compute m = mean of entries in normalized similarity */
        m = average_of_entries(W);

        /* Initialize H with values in [0, 2*sqrt(m/k)] */
        H_initilization(H, m, k);
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
        copy_matrix(dest, H);
    }
    if ((strcmp(strategy, "sym") == 0))
    {
        copy_matrix(dest, A);
    }
    else if ((strcmp(strategy, "ddg") == 0))
    {
        copy_matrix(dest, D);
    }
    else if ((strcmp(strategy, "norm") == 0))
    {
        copy_matrix(dest, W);
    }
    destroy_matrix(A);
    destroy_matrix(D);
    destroy_matrix(W);
    destroy_matrix(invsqrD_Mul_A);
    destroy_matrix(invsqrD);
    destroy_matrix(H);
    destroy_matrix(WH);
    destroy_matrix(H_transpose);
    destroy_matrix(HHT);
    destroy_matrix(HHTH);
    destroy_matrix(ratio_mat);
    destroy_matrix(beta_mul_and_addition);
    destroy_matrix(frob_sub);
    destroy_matrix(H_new);
    destroy_matrix(invsqrD_A);

    /*H_final = optimized decomposition matrix H*/
}

int main(int argc, char *argv[])
{
    int k = 1;
    int sym, ddg, norm, symnmf; /*boolean like for strategy*/
    char *filename = NULL, *strategy;
    Matrix X, mat_to_print;

    if (argc != 3 && argc != 4)
    {
        ERROR_EXIT();
    }
    sym = (strcmp(argv[1], "sym") == 0);
    ddg = (strcmp(argv[1], "ddg") == 0);
    norm = (strcmp(argv[1], "norm") == 0);
    symnmf = (strcmp(argv[1], "symnmf") == 0);

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
    read_mat_from_csv_file(filename, X);
    mat_to_print = symnmf ? create_matrix(X->rows, k) : create_matrix(X->rows, X->rows);
    if (sym)
    {

        strategy = "sym";
    }
    else if (ddg)
    {
        strategy = "ddg";
    }
    else if (norm)
    {
        strategy = "norm";
    }
    /* must be symnmf */
    else
    {
        strategy = "symnmf";
    }

    get_mat_of_strategy(mat_to_print, X, strategy, k);
    print_Matrix(mat_to_print);
    destroy_matrix(X);
    destroy_matrix(mat_to_print);
    return 0;
}
