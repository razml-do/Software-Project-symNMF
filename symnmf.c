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

double **allocate_matrix(int rows, int cols)
{
    int i;

    double **mat = malloc(sizeof(double *) * rows);
    if (!mat)
        ERROR_EXIT();

    for (i = 0; i < rows; i++)
    {
        mat[i] = calloc(cols, sizeof(double));
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
    double **ret = allocate_matrix(rows, cols);

    for (row = 0; row < rows; row++)
    {
        for (col = 0; col < cols; col++)
        {
            ret[row][col] = (mul * mat[row][col]) + add;
        }
    }
    return ret;
}

double **mat_transpose(double **mat, int rows, int cols)
{
    int row, col;
    double **ret = allocate_matrix(cols, rows);

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
    double **ret = allocate_matrix(rows, cols);

    for (row = 0; row < rows; row++)
    {
        for (col = 0; col < cols; col++)
        {
            ret[row][col] = mat1[row][col] - mat2[row][col];
        }
    }

    return ret;
}

double **mat_mul_indices(double **mat1, double **mat2, int rows, int cols)
{
    int row, col;
    double **ret = allocate_matrix(rows, cols);

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
    double **ret = allocate_matrix(rows, cols);

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
        {
            ret += mat[row][col] * mat[row][col];
        }
    }
    return ret;
}

double **similarity_matrix(double **mat, int d, int N)
{
    int row, col;
    double **ret = allocate_matrix(N, N);

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
    int i;
    double **ret = allocate_matrix(N, N);

    /*mat is diagonal, only operate on elements of the diagonal.*/
    for (i = 0; i < N; i++)
    {
        ret[i][i] = vertex_degree(mat[i], N);
    }

    return ret;
}

double **diagonal_inverse_square_root(double **mat, int N)
{
    int i;
    double **ret = allocate_matrix(N, N);

    /*diagonal matrix doesn't need to inverse sqrt on all elements,
    only operate on elements of the diagonal.*/
    for (i = 0; i < N; i++)
    {
        ret[i][i] = 1 / sqrt(mat[i][i]);
    }

    return ret;
}

double **mat_mul(double **mat1, int rows1, int cols1,
                 double **mat2, int rows2, int cols2)
{
    int i, j, k;
    double **ret = allocate_matrix(rows1, cols2);

    if (cols1 != rows2)
        ERROR_EXIT(); /* invalid dimensions */

    for (i = 0; i < rows1; i++)
    {
        for (j = 0; j < cols2; j++)
        {
            ret[i][j] = 0;
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

void free_matrix(double ***matrix, int N)
{
    int i;
    if (*matrix == NULL)
        return;

    for (i = 0; i < N; i++)
    {
        free((*matrix)[i]);
    }

    free(*matrix);
    *matrix = NULL;
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
    double **all[9];
    int count = 9, i, j;
    all[0] = dataPointArray;
    all[1] = A;
    all[2] = D_inverse_sqrt;
    all[3] = mat_to_print;
    all[4] = ddg_tmp;
    all[5] = tmp;
    all[6] = norm;
    all[7] = H;
    all[8] = W;

    for (i = 0; i < count; i++)
    {
        int already_freed = 0;
        for (j = 0; j < i; j++)
        {
            if (all[i] == all[j])
            {
                already_freed = 1;
                break;
            }
        }

        if (!already_freed && all[i])
        {
            free_matrix(&all[i], N);
        }
    }
}
void iterative_free(double **WH, double **H_transpose, double **HHT,
                    double **HHTH, double **ratio_mat,
                    double **beta_mul_and_addition, double **frob_sub,
                    int N, int k)
{
    free_matrix(&WH, N);
    free_matrix(&H_transpose, k);
    free_matrix(&HHT, N);
    free_matrix(&HHTH, N);
    free_matrix(&ratio_mat, N);
    free_matrix(&beta_mul_and_addition, N);
    free_matrix(&frob_sub, N);
}

/* -------------------- CSV READING -------------------- */

double **read_csv_file(const char *filename, int *rows_out, int *cols_out)
{
    char line[MAX_LINE_LEN];
    int rows = 0, cols = 0, t, i;
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
        {
            /* free already allocated rows */
            if (mat)
            {
                for (t = 0; t < rows; t++)
                    free(mat[t]);
                free(mat);
            }
            fclose(file);
            ERROR_EXIT();
        }

        token = strtok(line, ",");
        for (i = 0; i < cols; i++)
        {
            if (!token)
            {
                free(row_data);
                if (mat)
                {
                    for (t = 0; t < rows; t++)
                        free(mat[t]);
                    free(mat);
                }
                fclose(file);
                ERROR_EXIT();
            }
            row_data[i] = atof(token);
            token = strtok(NULL, ",");
        }

        tmp = realloc(mat, sizeof(double *) * (rows + 1));
        if (!tmp)
        {
            /* free row_data and previously allocated rows */
            free(row_data);
            if (mat)
            {
                for (t = 0; t < rows; t++)
                    free(mat[t]);
                free(mat);
            }
            fclose(file);
            ERROR_EXIT();
        }
        mat = tmp;
        mat[rows] = row_data;
        rows++;
    }

    fclose(file);
    *rows_out = rows;
    *cols_out = cols;
    return mat;
}

double **run_symnmf(double **H, double **W, int N, int k)
{
    int i;
    const int max_iter = 300;
    const double beta = 0.5, epsilon = 1e-4;
    double **WH = NULL, **H_transpose = NULL, **HHT = NULL, **HHTH = NULL;
    double **ratio_mat = NULL, **beta_mul_and_addition = NULL, **frob_sub = NULL;
    double **H_new = NULL;
    double squared_frob;

    for (i = 0; i < max_iter; i++)
    {
        /*
        TODO: dont reallocate matrices in every iteration.
              allocate once at the top once and free at end of function.
        */
        WH = mat_mul(W, N, N, H, N, k);
        H_transpose = mat_transpose(H, N, k);
        HHT = mat_mul(H, N, k, H_transpose, k, N);
        HHTH = mat_mul(HHT, N, N, H, N, k);
        ratio_mat = mat_div(WH, HHTH, N, k);
        beta_mul_and_addition = mat_const_mul_and_add(ratio_mat, N, k, beta, 1 - beta);

        H_new = mat_mul_indices(H, beta_mul_and_addition, N, k);
        frob_sub = mat_sub(H_new, H, N, k);
        squared_frob = frobenius_norm_squared(frob_sub, N, k);

        iterative_free(WH, H_transpose, HHT, HHTH, ratio_mat, beta_mul_and_addition, frob_sub, N, k);

        if (H)
        {
            free_matrix(&H, N);
        }

        H = H_new;
        H_new = NULL;

        if (squared_frob < epsilon)
            break;
    }

    return H;
}

int main(int argc, char *argv[])
{
    int N = 0, d = 0, k = 0;
    double m = 0;
    double **X = NULL;
    double **sim = NULL;
    double **D = NULL;
    double **W = NULL;
    double **norm_mat = NULL;
    double **H = NULL;
    double **H_final = NULL;
    double **inverseSqrtD_Mul_A = NULL;
    int i, j;

    if (strcmp(argv[1], "sym") == 0)
    {
        if (argc < 3)
        {
            ERROR_EXIT(); /*not enough parameters.*/
        }

        /* Read input CSV file */
        X = read_csv_file(argv[2], &N, &d);
        if (!X)
        {
            ERROR_EXIT(); /*couldn't read file*/
        }

        sim = similarity_matrix(X, d, N);
        if (!sim)
            ERROR_EXIT();

        print_matrix(sim, N, N);
        free_all(X, sim, NULL, NULL, NULL, NULL, NULL, NULL, NULL, N);
    }
    else if (strcmp(argv[1], "ddg") == 0)
    {
        /* Read input CSV file */
        X = read_csv_file(argv[2], &N, &d);
        if (!X)
        {
            ERROR_EXIT(); /*couldn't read file*/
        }

        sim = similarity_matrix(X, d, N);
        if (!sim)
            ERROR_EXIT();

        D = diagonal_degree_matrix(sim, N);
        print_matrix(D, N, N);
        free_all(X, sim, D, NULL, NULL, NULL, NULL, NULL, NULL, N);
    }
    else if (strcmp(argv[1], "norm") == 0)
    {
        /* Read input CSV file */
        X = read_csv_file(argv[2], &N, &d);
        if (!X)
        {
            ERROR_EXIT(); /*couldn't read file*/
        }

        sim = similarity_matrix(X, d, N);
        if (!sim)
            ERROR_EXIT();

        D = diagonal_degree_matrix(sim, N);
        W = diagonal_inverse_square_root(D, N);

        norm_mat = mat_mul(W, N, N, sim, N, N);
        inverseSqrtD_Mul_A = mat_mul(norm_mat, N, N, W, N, N);
        free_matrix(&norm_mat, N);
        norm_mat = inverseSqrtD_Mul_A;

        print_matrix(norm_mat, N, N);
        free_all(X, sim, D, W, norm_mat, NULL, NULL, NULL, NULL, N);
    }
    else if (strcmp(argv[1], "symnmf") == 0)
    {
        if (argc != 4)
            ERROR_EXIT();

        /* Read input CSV file */
        X = read_csv_file(argv[3], &N, &d);
        if (!X)
        {
            ERROR_EXIT(); /*couldn't read file*/
        }

        k = atoi(argv[2]);
        if (k <= 0 || k > N)
            ERROR_EXIT();

        /* Compute similarity matrix and normalized similarity */
        sim = similarity_matrix(X, d, N);
        if (!sim)
            ERROR_EXIT();

        D = diagonal_degree_matrix(sim, N);
        W = diagonal_inverse_square_root(D, N);

        norm_mat = mat_mul(W, N, N, sim, N, N);                /*D^(-0.5) * SIM*/
        inverseSqrtD_Mul_A = mat_mul(norm_mat, N, N, W, N, N); /*D^(-0.5) * SIM * D^(-0.5)*/
        free_matrix(&norm_mat, N);                             /*delete first matrix for mat mul*/
        norm_mat = inverseSqrtD_Mul_A;                         /*assign new matrix*/

        /* Compute m = mean of entries in normalized similarity */
        m = 0.0;
        for (i = 0; i < N; i++)
            for (j = 0; j < N; j++)
                m += norm_mat[i][j];
        m /= (N * N);

        /* Initialize H with values in [0, 2*sqrt(m/k)] */
        H = allocate_matrix(N, k);
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < k; j++)
            {
                H[i][j] = ((double)rand() / RAND_MAX) * 2 * sqrt(m / k);
            }
        }

        /* Run SymNMF */
        H_final = run_symnmf(H, norm_mat, N, k);
        if (!H_final)
        {
            ERROR_EXIT();
        }

        print_matrix(H_final, N, k);

        free_matrix(&sim, N);
        free_matrix(&D, N);
        free_matrix(&W, N);
        free_matrix(&norm_mat, N);
        /*free_matrix(&H, N); H is freed inside run_symnmf, no need to free again.*/
        free_matrix(&H_final, N);
    }
    else
    {
        free_matrix(&X, N);
        printf("An Error Has Occurred\n");
        return 1;
    }

    return 0;
}