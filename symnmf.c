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

double squared_euclidean_distance(double *p, double *q, int d)
{
    int i;
    double squareSum = 0;

    for (i = 0; i < d; i++)
        squareSum += (p[i] - q[i]) * (p[i] - q[i]);

    return squareSum;
}

double **mat_const_mul_and_add(double **mat, int rows, int colms, double mul, double add)
{
    int row, col;
    double **ret = NULL;

    ret = (double **)malloc(sizeof(double *) * rows);
    if (!ret)
        ERROR_EXIT();

    for (row = 0; row < rows; row++)
    {
        ret[row] = (double *)malloc(colms * sizeof(double));
        if (!ret[row])
            ERROR_EXIT();
        for (col = 0; col < colms; col++)
            ret[row][col] = (mul * mat[row][col]) + add;
    }
    return ret;
}

double **mat_transpose(double **mat, int rows, int colms)
{
    int row, col;
    double **ret = NULL;

    ret = (double **)malloc(sizeof(double *) * colms);
    if (!ret)
        ERROR_EXIT();

    for (col = 0; col < colms; col++)
    {
        ret[col] = (double *)malloc(rows * sizeof(double));
        if (!ret[col])
            ERROR_EXIT();
    }
    for (row = 0; row < rows; row++)
    {
        for (col = 0; col < colms; col++)
            ret[col][row] = mat[row][col];
    }
    return ret;
}

double **mat_sub(double **mat1, double **mat2, int rows, int colms)
{
    int row, col;
    double **ret = NULL;

    ret = (double **)malloc(sizeof(double *) * rows);
    if (!ret)
        ERROR_EXIT();

    for (row = 0; row < rows; row++)
    {
        ret[row] = (double *)malloc(colms * sizeof(double));
        if (!ret[row])
            ERROR_EXIT();
        for (col = 0; col < colms; col++)
            ret[row][col] = mat1[row][col] - mat2[row][col];
    }
    return ret;
}

double **mat_mul_indices(double **mat1, double **mat2, int rows, int colms)
{
    int row, col;
    double **ret = NULL;

    ret = (double **)malloc(sizeof(double *) * rows);
    if (!ret)
        ERROR_EXIT();

    for (row = 0; row < rows; row++)
    {
        ret[row] = (double *)malloc(colms * sizeof(double));
        if (!ret[row])
            ERROR_EXIT();
        for (col = 0; col < colms; col++)
        {
            if (mat2[row][col] == 0)
                ERROR_EXIT();
            ret[row][col] = mat1[row][col] * mat2[row][col];
        }
    }
    return ret;
}

double **mat_div(double **mat1, double **mat2, int rows, int colms)
{
    int row, col;
    double **ret = NULL;

    ret = (double **)malloc(sizeof(double *) * rows);
    if (!ret)
        ERROR_EXIT();

    for (row = 0; row < rows; row++)
    {
        ret[row] = (double *)malloc(colms * sizeof(double));
        if (!ret[row])
            ERROR_EXIT();
        for (col = 0; col < colms; col++)
        {
            if (mat2[row][col] == 0)
                ERROR_EXIT();
            ret[row][col] = mat1[row][col] / mat2[row][col];
        }
    }
    return ret;
}

double frobenius_norm_squared(double **mat, int rows, int colms)
{
    int row, col;
    double ret = 0;
    for (row = 0; row < rows; row++)
    {
        for (col = 0; col < colms; col++)
            ret += mat[row][col] * mat[row][col];
    }
    return ret;
}

double **similarity_matrix(double **mat, int d, int N)
{
    int row, col;
    double **ret = NULL;

    ret = (double **)malloc(sizeof(double *) * N);
    if (!ret)
        ERROR_EXIT();

    for (row = 0; row < N; row++)
    {
        ret[row] = (double *)calloc(N, sizeof(double));

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
    double **ret = (double **)malloc(sizeof(double *) * N);
    if (!ret)
        ERROR_EXIT();
    for (i = 0; i < N; i++)
    {
        ret[i] = (double *)malloc(sizeof(double) * N);
        if (!ret[i])
            ERROR_EXIT();

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
    double **ret = (double **)malloc(sizeof(double *) * N);
    if (!ret)
        ERROR_EXIT();
    for (i = 0; i < N; i++)
    {
        ret[i] = (double *)malloc(sizeof(double) * N);
        if (!ret[i])
            ERROR_EXIT();
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
    double **ret;

    if (cols1 != rows2)
        ERROR_EXIT(); /* invalid dimensions */

    ret = (double **)malloc(sizeof(double *) * rows1);
    if (!ret)
        ERROR_EXIT();

    for (i = 0; i < rows1; i++)
    {
        ret[i] = (double *)calloc(cols2, sizeof(double));
        if (!ret[i])
            ERROR_EXIT();

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

void free_all(char *line1,
              char *line2,
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
    if (line1)
        free(line1);
    if (line2)
        free(line2);
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
void iterative_free(double **WH, double **H_transpose, double **HHT, double **HHTH, double **ratio_mat, double **beta_mul_and_addition, double **H_new, double **frob_sub, int rows)
{
    free_matrix(&WH, rows);
    free_matrix(&H_transpose, rows);
    free_matrix(&HHT, rows);
    free_matrix(&HHTH, rows);
    free_matrix(&ratio_mat, rows);
    free_matrix(&beta_mul_and_addition, rows);
    free_matrix(&H_new, rows);
    free_matrix(&frob_sub, rows);
}

/* -------------------- CSV READING -------------------- */

double **read_csv_file(const char *filename, int *rows_out, int *cols_out)
{
    FILE *file = fopen(filename, "r");
    if (!file)
        ERROR_EXIT();

    char line[MAX_LINE_LEN];
    int rows = 0, cols = 0;
    double **mat = NULL;

    while (fgets(line, sizeof(line), file))
    {
        if (cols == 0)
            cols = count_commas(line) + 1;
        double *row_data = malloc(sizeof(double) * cols);
        if (!row_data)
            ERROR_EXIT();

        char *token = strtok(line, ",");
        for (int i = 0; i < cols; i++)
        {
            if (!token)
                ERROR_EXIT();
            row_data[i] = atof(token);
            token = strtok(NULL, ",");
        }

        double **tmp = realloc(mat, sizeof(double *) * (rows + 1));
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
    char *strategy = NULL, *token = NULL, *line1 = NULL, *line2 = NULL;
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

    file = fopen(argv[2], "r");
    if (!file)
        ERROR_EXIT();

    line1 = malloc(sizeof(char) * MAX_LINE_LEN);
    line2 = malloc(sizeof(char) * MAX_LINE_LEN);
    if (!line1 || !line2)
        ERROR_EXIT();

    while (fgets(line1, MAX_LINE_LEN, file))
    {
        if (d == 0)
            d = count_commas(line1) + 1;

        dataPoint1 = malloc(sizeof(double) * d);
        if (!dataPoint1)
            ERROR_EXIT();

        token = strtok(line1, ",");
        if (!token)
            ERROR_EXIT();

        for (i = 0; i < d; i++)
        {
            dataPoint1[i] = atof(token);
            token = strtok(NULL, ",");
        }

        N++;
        tmp = realloc(dataPointArray, sizeof(double *) * N);
        if (!tmp)
            ERROR_EXIT();

        dataPointArray = tmp;
        tmp = NULL;
        dataPointArray[N - 1] = dataPoint1;
    }
    fclose(file);
    if (strcmp(strategy, "symnmf") == 0)
    {
        fileH = fopen(argv[3], "r");
        fileW = fopen(argv[4], "r");
        k = atoi(argv[5]);
        H = (double **)malloc(sizeof(double *) * N);
        W = (double **)malloc(sizeof(double *) * N);
        if (!H || !W)
            ERROR_EXIT();
        row_symnmf = 0;
        while (fgets(line1, MAX_LINE_LEN, fileH) && fgets(line2, MAX_LINE_LEN, fileW))
        {
            dataPoint1 = malloc(sizeof(double) * k);
            dataPoint2 = malloc(sizeof(double) * N);
            if (!dataPoint1 || !dataPoint2)
                ERROR_EXIT();

            token = strtok(line1, ",");
            if (!token)
                ERROR_EXIT();
            for (i = 0; i < k; i++)
            {
                dataPoint1[i] = atof(token);
                token = strtok(NULL, ",");
            }
            token = strtok(line2, ",");
            if (!token)
                ERROR_EXIT();
            for (i = 0; i < N; i++)
            {
                dataPoint2[i] = atof(token);
                token = strtok(NULL, ",");
            }

            H[row_symnmf] = dataPoint1;
            W[row_symnmf] = dataPoint2;
            dataPoint1 = NULL;
            dataPoint2 = NULL;
            row_symnmf++;
        }
        fclose(fileH);
        fclose(fileW);
    }
    if (N == 0)
    {
        free_all(line1, line2, dataPointArray, A, D_inverse_sqrt, mat_to_print, ddg_tmp, tmp, norm, H, W, N);
        return 1;
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
        WH = mat_mul(W, N, N, H, N, k);
        H_transpose = mat_transpose(H, N, k);
        HHT = mat_mul(H, N, k, H_transpose, k, N);
        HHTH = mat_mul(HHT, N, N, H, k, N);
        ratio_mat = mat_div(WH, HHTH, N, k);
        beta_mul_and_addition = mat_const_mul_and_add(ratio_mat, N, k, beta, 1 - beta);
        H_new = mat_mul_indices(H, beta_mul_and_addition, N, k);
        frob_sub = mat_sub(H_new, H, N, k);
        for (i = 0; i < max_iter; i++)
        {
            squared_frob = frobenius_norm_squared(frob_sub, N, k);
            if (squared_frob < epsilon)
                break;
            iterative_free(WH, H_transpose, HHT, HHTH, ratio_mat, beta_mul_and_addition, H_new, frob_sub, N);
            tmp_H = H;
            H = H_new;
            free_matrix(&tmp_H, N);
            WH = mat_mul(W, N, N, H, N, k);
            H_transpose = mat_transpose(H, N, k);
            HHT = mat_mul(H, N, k, H_transpose, k, N);
            HHTH = mat_mul(HHT, N, N, H, k, N);
            ratio_mat = mat_div(WH, HHTH, N, k);
            beta_mul_and_addition = mat_const_mul_and_add(ratio_mat, N, k, beta, 1 - beta);
            H_new = mat_mul_indices(H, beta_mul_and_addition, N, k);
            frob_sub = mat_sub(H_new, H, N, k);
        }
        iterative_free(WH, H_transpose, HHT, HHTH, ratio_mat, beta_mul_and_addition, H_new, frob_sub, N);
        mat_to_print = H;
    }
    print_matrix(mat_to_print, N, N);
    free_all(line1, line2, dataPointArray, A, D_inverse_sqrt, mat_to_print, ddg_tmp, tmp, norm, H, W, N);
    return 0;
}
