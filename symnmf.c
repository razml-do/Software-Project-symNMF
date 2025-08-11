#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_LINE_LEN 4096

void free_all(void);
#define ERROR_EXIT()                      \
    do                                    \
    {                                     \
        printf("An Error Has Occured\n"); \
        printf("Line: %d\n", __LINE__);   \
        free_all();                       \
        exit(1);                          \
    } while (0)

FILE *file;
char *strategy = NULL, *token = NULL, *line = NULL;
double *dataPoint = NULL;
double **dataPointArray = NULL, **D_inverse_sqrt = NULL, **A = NULL, **mat_to_print = NULL, **ddg_tmp = NULL, **tmp = NULL;
int N = 0, d = 0, i;

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
    {
        squareSum += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return squareSum;
}

double **similarity_matrix(double **mat, int d, int N)
{
    int i, j;
    double **ret = NULL;
    ret = (double **)malloc(sizeof(double *) * N);
    if (!ret)
        ERROR_EXIT();
    for (i = 0; i < N; i++)
    {
        ret[i] = (double *)malloc(sizeof(double) * N);
        for (j = 0; j < N; j++)
        {
            ret[i][j] = 0;
            if (i != j)
                ret[i][j] = exp(-squared_euclidean_distance(mat[i], mat[j], d) / 2.0);
        }
    }
    return ret;
}

double vertex_degree(double *row, int d)
{
    double deg = 0;
    int i;
    for (i = 0; i < d; i++)
    {
        deg += row[i];
    }
    return deg;
}

double **diagonal_degree_matrix(double **mat, int d, int N)
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
                ret[i][j] = vertex_degree(mat[i], d);
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

double **mat_mul(double **mat1, double **mat2, int N)
{
    int i, j, z;
    double **ret = (double **)malloc(sizeof(double *) * N);
    if (!ret)
        ERROR_EXIT();
    for (i = 0; i < N; i++)
    {
        ret[i] = (double *)calloc(N, sizeof(double));
        if (!ret[i])
            ERROR_EXIT();
        for (j = 0; j < N; j++)
        {
            for (z = 0; z < N; z++)
                ret[i][j] += mat1[i][z] * mat2[z][j];
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

void free_matrix(double ***mat_ptr)
{
    int i;
    if (!(*mat_ptr))
        return;
    for (i = 0; i < N; i++)
    {
        if ((*mat_ptr)[i])
            free((*mat_ptr)[i]);
    }
    free(*mat_ptr);
    *mat_ptr = NULL;
}

void free_all()
{
    int i;
    if (line)
    {
        free(line);
        line = NULL;
    }

    if (dataPointArray)
    {
        if (N > 0)
        {
            for (i = 0; i < N; i++)
            {
                if (dataPointArray[i])
                    free(dataPointArray[i]);
            }
        }
        free(dataPointArray);
        dataPointArray = NULL;
    }

    free_matrix(&A);
    free_matrix(&D_inverse_sqrt);
    free_matrix(&mat_to_print);
    free_matrix(&ddg_tmp);
    free_matrix(&tmp);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
        ERROR_EXIT();
    strategy = argv[1];

    if (strcmp(strategy, "sym") != 0 && strcmp(strategy, "ddg") != 0 && strcmp(strategy, "norm") != 0)
        ERROR_EXIT();

    file = fopen(argv[2], "r");
    if (!file)
        ERROR_EXIT();

    line = malloc(sizeof(char) * MAX_LINE_LEN);
    if (!line)
        ERROR_EXIT();
    while (fgets(line, MAX_LINE_LEN, file))
    {
        if (d == 0)
            d = count_commas(line) + 1;
        dataPoint = malloc(sizeof(double) * d);
        if (!dataPoint)
            ERROR_EXIT();
        token = strtok(line, ",");
        for (i = 0; i < d; i++)
        {
            dataPoint[i] = atof(token);
            token = strtok(NULL, ",");
        }
        N++;
        tmp = realloc(dataPointArray, sizeof(double *) * N);
        if (!tmp)
            ERROR_EXIT();
        dataPointArray = tmp;
        dataPointArray[N - 1] = dataPoint;
    }
    fclose(file);

    if (strcmp(strategy, "sym") == 0)
        mat_to_print = similarity_matrix(dataPointArray, d, N);

    if (strcmp(strategy, "ddg") == 0)
        mat_to_print = diagonal_degree_matrix(dataPointArray, d, N);

    if (strcmp(strategy, "norm") == 0)
    {
        A = similarity_matrix(dataPointArray, d, N);
        ddg_tmp = diagonal_degree_matrix(dataPointArray, d, N);
        D_inverse_sqrt = diagonal_inverse_square_root(ddg_tmp, N);
        tmp = mat_mul(D_inverse_sqrt, A, N);
        mat_to_print = mat_mul(tmp, D_inverse_sqrt, N);
    }

    print_matrix(mat_to_print, N, N);
    free_all();
    return 0;
}
