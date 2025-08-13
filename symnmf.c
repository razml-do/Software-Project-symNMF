#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_LINE_LEN 4096

#define ERROR_EXIT()                      \
    do                                    \
    {                                     \
        printf("An Error Has Occured\n"); \
        exit(1);                          \
    } while (0)

/* -------------------- UTILITY FUNCTIONS -------------------- */
void print_matrix(double **mat, int rows, int cols)
{
    int i, j;
    for (i = 0; i < rows; i++)
    {
        if (!mat[i])
            continue;

        for (j = 0; j < cols; j++)
        {
            printf("%.4f", mat[i][j]);
            if (j < cols - 1)
                printf(",");
        }

        printf("\n");
    }
}

double **allocate_matrix(int rows, int cols, int zero_init)
{
    double **mat = malloc(sizeof(double *) * rows);
    if (!mat)
        ERROR_EXIT();
    for (int i = 0; i < rows; i++)
    {
        mat[i] = zero_init ? calloc(cols, sizeof(double)) : malloc(sizeof(double) * cols);
        if (!mat[i])
            ERROR_EXIT();
    }
    return mat;
}

double squared_euclidean_distance(double *p, double *q, int d)
{
    int i;
    double squareSum = 0;

    for (i = 0; i < d; i++)
        squareSum += (p[i] - q[i]) * (p[i] - q[i]);

    return squareSum;
}

double **mat_const_mul_and_add(double **mat, int rows, int cols, double mul, double add)
{
    int row, col;
    double **ret = allocate_matrix(rows, cols, 0);

    for (row = 0; row < rows; row++)
    {
        for (col = 0; col < cols; col++)
            ret[row][col] = (mul * mat[row][col]) + add;
    }
    return ret;
}

double **mat_transpose(double **mat, int rows, int cols)
{
    int row, col;
    double **ret = allocate_matrix(cols, rows, 0);

    for (row = 0; row < rows; row++)
    {
        for (col = 0; col < cols; col++)
            ret[col][row] = mat[row][col];
    }
    return ret;
}

double **mat_sub(double **mat1, double **mat2, int rows, int cols)
{
    int row, col;
    double **ret = allocate_matrix(rows, cols, 0);

    for (row = 0; row < rows; row++)
    {
        for (col = 0; col < cols; col++)
            ret[row][col] = mat1[row][col] - mat2[row][col];
    }
    return ret;
}

double **mat_mul_indices(double **mat1, double **mat2, int rows, int cols)
{
    int row, col;
    double **ret = allocate_matrix(rows, cols, 0);

    for (row = 0; row < rows; row++)
    {
        for (col = 0; col < cols; col++)
        {
            if (mat2[row][col] == 0)
                ERROR_EXIT();
            ret[row][col] = mat1[row][col] * mat2[row][col];
        }
    }
    return ret;
}

double **mat_div(double **mat1, double **mat2, int rows, int cols)
{
    int row, col;
    double **ret = allocate_matrix(rows, cols, 0);

    for (row = 0; row < rows; row++)
    {
        for (col = 0; col < cols; col++)
        {
            if (mat2[row][col] == 0)
                ERROR_EXIT();
            ret[row][col] = mat1[row][col] / mat2[row][col];
        }
    }
    return ret;
}

double frobenius_norm_squared(double **mat, int rows, int cols)
{
    int row, col;
    double ret = 0;
    for (row = 0; row < rows; row++)
    {
        for (col = 0; col < cols; col++)
            ret += mat[row][col] * mat[row][col];
    }
    return ret;
}

double **similarity_matrix(double **mat, int d, int N)
{
    int row, col;
    double **ret = allocate_matrix(N, N, 0);

    for (row = 0; row < N; row++)
    {
        for (col = 0; col < N; col++)
        {
            if (row != col)
                ret[row][col] = exp(-squared_euclidean_distance(mat[row], mat[col], d) / 2.0);
        }
    }
    return ret;
}

double vertex_degree(double *row, int d)
{
    double deg = 0;
    int i;
    for (i = 0; i < d; i++)
        deg += row[i];
    return deg;
}

double **diagonal_degree_matrix(double **mat, int N)
{
    int i, j;
    double **ret = allocate_matrix(N, N, 0);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            ret[i][j] = 0;
            if (i == j)
                ret[i][j] = vertex_degree(mat[i], N);
        }
    }
    return ret;
}

double **diagonal_inverse_square_root(double **mat, int N)
{
    int i, j;
    double **ret = allocate_matrix(N, N, 0);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            ret[i][j] = 0;
            if (i == j)
                ret[i][j] = 1 / sqrt(mat[i][j]);
        }
    }
    return ret;
}

double **mat_mul(double **mat1, int rows1, int cols1,
                 double **mat2, int rows2, int cols2)
{
    int i, j, k;
    double **ret = allocate_matrix(rows1, cols2, 0);

    if (cols1 != rows2)
        ERROR_EXIT(); /* invalid dimensions */

    for (i = 0; i < rows1; i++)
    {
        for (j = 0; j < cols2; j++)
        {
            for (k = 0; k < cols1; k++)
            {
                ret[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return ret;
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

void free_matrix(double ***mat_ptr, int rows)
{
    int i;

    if (!mat_ptr || !(*mat_ptr))
    {
        return;
    }

    for (i = 0; i < rows; i++)
    {
        if ((*mat_ptr)[i])
        {
            free((*mat_ptr)[i]);
        }
    }

    free(*mat_ptr);
    *mat_ptr = NULL;
}

void free_all(
    double **dataPointArray,
    double **A,
    double **D_inverse_sqrt,
    double **mat_to_print,
    double **ddg_tmp,
    double **tmp,
    double **norm,
    double **H,
    double **W,
    int N)
{
    free_matrix(&dataPointArray, N);
    free_matrix(&A, N);
    free_matrix(&D_inverse_sqrt, N);
    free_matrix(&mat_to_print, N);
    free_matrix(&ddg_tmp, N);
    free_matrix(&tmp, N);
    free_matrix(&norm, N);
    free_matrix(&H, N);
    free_matrix(&W, N);
}
void iterative_free(double **WH, double **H_transpose, double **HHT, double **HHTH, double **ratio_mat, double **beta_mul_and_addition, double **frob_sub, int rows)
{
    free_matrix(&WH, rows);
    free_matrix(&H_transpose, rows);
    free_matrix(&HHT, rows);
    free_matrix(&HHTH, rows);
    free_matrix(&ratio_mat, rows);
    free_matrix(&beta_mul_and_addition, rows);
    free_matrix(&frob_sub, rows);
}

/* -------------------- CSV READING -------------------- */

double **read_csv_file(const char *filename, int *rows_out, int *cols_out)
{
    char line[MAX_LINE_LEN];
    int rows = 0, cols = 0;
    double **mat = NULL;
    double *row_data;
    char *token;
    double **tmp;
    FILE *file = fopen(filename, "r");
    if (!file)
        ERROR_EXIT();

    while (fgets(line, sizeof(line), file))
    {
        if (cols == 0)
            cols = count_commas(line) + 1;
        row_data = malloc(sizeof(double) * cols);
        if (!row_data)
            ERROR_EXIT();

        token = strtok(line, ",");
        for (int i = 0; i < cols; i++)
        {
            if (!token)
                ERROR_EXIT();
            row_data[i] = atof(token);
            token = strtok(NULL, ",");
        }

        tmp = realloc(mat, sizeof(double *) * (rows + 1));
        if (!tmp)
            ERROR_EXIT();
        mat = tmp;
        mat[rows] = row_data;
        rows++;
    }

    fclose(file);
    *rows_out = rows;
    *cols_out = cols;
    return mat;
}

int main(int argc, char *argv[])
{
    FILE *file, *fileH, *fileW;
    char *strategy = NULL, *token = NULL;
    double squared_frob;
    double *dataPoint1 = NULL, *dataPoint2 = NULL;
    double **dataPointArray = NULL, **D_inverse_sqrt = NULL, **A = NULL, **mat_to_print = NULL, **ddg_tmp = NULL, **tmp = NULL, **norm = NULL, **H = NULL, **W = NULL, **H_new = NULL, **WH = NULL, **H_transpose = NULL, **HHT = NULL, **HHTH = NULL, **ratio_mat = NULL, **beta_mul_and_addition = NULL, **frob_sub = NULL, **tmp_H = NULL;
    int N = 0, d = 0, i, k, row_symnmf;
    const int max_iter = 300;
    const double beta = 0.5, epsilon = 1e-4;

    if (argc != 3 && argc != 6)
        ERROR_EXIT();

    strategy = argv[1];

    if (strcmp(strategy, "sym") != 0 && strcmp(strategy, "ddg") != 0 && strcmp(strategy, "norm") != 0 && strcmp(strategy, "symnmf") != 0)
        ERROR_EXIT();

    dataPointArray = read_csv_file(argv[2], &N, &d);
    if (N == 0)
    {
        free_all(dataPointArray, A, D_inverse_sqrt, mat_to_print, ddg_tmp, tmp, norm, H, W, N);
        return 1;
    }

    if (strcmp(strategy, "symnmf") == 0)
    {
        k = atoi(argv[5]);
        H = read_csv_file(argv[3], &N, &k);
        W = read_csv_file(argv[4], &N, &N);
    }

    A = similarity_matrix(dataPointArray, d, N);
    ddg_tmp = diagonal_degree_matrix(A, N);
    D_inverse_sqrt = diagonal_inverse_square_root(ddg_tmp, N);
    tmp = mat_mul(D_inverse_sqrt, N, N, A, N, N);
    norm = mat_mul(tmp, N, N, D_inverse_sqrt, N, N);
    if (strcmp(strategy, "sym") == 0)
        mat_to_print = A;
    if (strcmp(strategy, "ddg") == 0)
        mat_to_print = ddg_tmp;
    if (strcmp(strategy, "norm") == 0)
        mat_to_print = norm;
    else
    {
        for (i = 0; i < max_iter; i++)
        {
            WH = mat_mul(W, N, N, H, N, k);
            H_transpose = mat_transpose(H, N, k);
            HHT = mat_mul(H, N, k, H_transpose, k, N);
            HHTH = mat_mul(HHT, N, N, H, k, N);
            ratio_mat = mat_div(WH, HHTH, N, k);
            beta_mul_and_addition = mat_const_mul_and_add(ratio_mat, N, k, beta, 1 - beta);
            H_new = mat_mul_indices(H, beta_mul_and_addition, N, k);
            frob_sub = mat_sub(H_new, H, N, k);
            squared_frob = frobenius_norm_squared(frob_sub, N, k);
            iterative_free(WH, H_transpose, HHT, HHTH, ratio_mat, beta_mul_and_addition, frob_sub, N);
            if (squared_frob < epsilon)
                break;
            tmp_H = H;
            H = H_new;
            free_matrix(&tmp_H, N);
        }
        if (H_new != H && H_new)
            free_matrix(&H_new, N);
        mat_to_print = H;
    }
    print_matrix(mat_to_print, N, N);
    free_all(dataPointArray, A, D_inverse_sqrt, mat_to_print, ddg_tmp, tmp, norm, H, W, N);
    return 0;
}
