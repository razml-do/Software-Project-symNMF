#ifndef SYM_NMF_H
#define SYM_NMF_H

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

/* Matrix allocation and deallocation */
matrix create_matrix(int rows, int cols);
void destroy_matrix(matrix *M);
void copy_matrix(matrix dest, matrix src);
void matrix_add_row(matrix *M);

/* Utility functions */
void print_Matrix(matrix M);
double squared_euclidean_distance(double *p, double *q, int d);
double frobenius_norm_squared(matrix M);
double vertex_degree(double *row, int d);
double average_of_entries(matrix W);
int count_commas(const char *line);

/* Matrix operations */
void mat_const_mul_and_add(matrix dest, matrix M, double mul, double add);
void mat_transpose(matrix dest, matrix M);
void mat_sub(matrix dest, matrix M1, matrix M2);
void mat_mul_indices(matrix dest, matrix M1, matrix M2);
void mat_div(matrix dest, matrix M1, matrix M2);
void mat_mul(matrix dest, matrix M1, matrix M2);

/* Graph / similarity related */
void diagonal_degree_matrix(matrix dest, matrix sim);
void similarity_matrix(matrix dest, matrix X);
void diagonal_inverse_square_root(matrix dest, matrix D);

/* H initialization */
void H_initilization(matrix H, double m, int k);

/* Strategy functions */
void sim_strategy(matrix dest, matrix X);
void ddg_strategy(matrix dest, matrix X);
void norm_strategy(matrix dest, matrix X);
void symnmf_strategy(matrix dest, int k, matrix W, matrix H);

/* Higher level function */
matrix get_mat_of_strategy(matrix X, char *strategy, int k);

/* CSV reading */
void read_mat_from_csv_file(const char *filename, matrix *M);

#endif /* SYM_NMF_H */
