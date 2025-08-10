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
        printf("Line: %d\n", __LINE__);   \
        free_all();                       \
        return 1;                         \
    } while (0)

#pragma region Variables
FILE *file;
char *strategy;
char *saveptr = NULL;
char *endptr;
char *token;
char *line;
char *split;
int **clusterGroupArr = NULL;
int *clusterGroupArrSizes = NULL;
double **dataPointArray = NULL;
double *sigmaDataPoint = NULL;
double *dataPoint = NULL;
double **uArr = NULL;
double **uPreviousArr = NULL;
double *tmp;
double epsilon = 0.001, minDistance, tempDistance;
int k, iter, N = 0, d = 0, i, j, z, iteration_number = 0, minCentroid;
size_t bufsize = 0;
int isFirstIteration = 1;
ssize_t linelen;
double coordinate;
#pragma endregion

#pragma region Methods

void print_matrix(double **mat, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        if (!mat[i])
            continue;
        for (int j = 0; j < cols; j++)
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

int count_commas(const char *line)
{
    int count = 0;
    for (int i = 0; line[i] != '\0'; i++)
    {
        if (line[i] == ',')
        {
            count++;
        }
    }
    return count;
}

void free_all(void)
{
}
#pragma endregion

int main(int argc, char *argv[])
{
    // Start Checks
    if (argc != 3)
        ERROR_EXIT();
    strategy = argv[1];

    if (strcmp(strategy, "sym") != 0 && strcmp(strategy, "ddg") != 0 && strcmp(strategy, "norm") != 0)
        ERROR_EXIT();

    file = fopen(argv[2], "r");
    if (!file)
        ERROR_EXIT();

    // File to dataPointArray
    line = malloc(sizeof(int) * MAX_LINE_LEN);
    while (fgets(line, MAX_LINE_LEN, file))
    {
        if (d == 0)
            d = count_commas(line) + 1;
        dataPoint = malloc(sizeof(double) * d);
        token = strtok(line, ",");
        for (i = 0; i < d; i++)
        {
            dataPoint[i] = atof(token);
            token = strtok(NULL, ",");
        }
        N++;
        dataPointArray = (double **)realloc(dataPointArray, sizeof(double *) * N);
        dataPointArray[N - 1] = dataPoint;
    }

#pragma region Similarity
    if (strcmp(strategy, "sym") == 0)
    {
        print_matrix(similarity_matrix(dataPointArray, d, N), N, N);
    }
#pragma endregion

#pragma region Diagonal Degree
    if (strcmp(strategy, "ddg") == 0)
    {
    }
#pragma endregion

#pragma region normalized
    if (strcmp(strategy, "norm") == 0)
    {
    }
#pragma endregion
}
