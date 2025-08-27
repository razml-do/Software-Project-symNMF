#ifndef SYM_NMF_H
#define SYM_NMF_H

double **read_csv_file(const char *filename, int *rows, int *cols);
double **similarity_matrix(double **data, int d, int N);
double **diagonal_degree_matrix(double **A, int N);
double **diagonal_inverse_square_root(double **D, int N);

double **mat_mul(double **A, int A_rows, int A_cols, double **B, int B_rows, int B_cols);
double **mat_transpose(double **A, int rows, int cols);
double **mat_div(double **A, double **B, int rows, int cols);
double **mat_const_mul_and_add(double **A, int rows, int cols, double mul, double add);
double **mat_mul_indices(double **H, double **B, int rows, int cols);
double **mat_sub(double **A, double **B, int rows, int cols);
double frobenius_norm_squared(double **A, int rows, int cols);

void iterative_free(double **WH, double **H_transpose, double **HHT, double **HHTH,
                    double **ratio_mat, double **beta_mul_and_addition, double **frob_sub,
                    int rows, int cols);
void free_matrix(double ***mat, int rows);
void free_all(double **data, double **A, double **B, double **C, double **D,
              double **E, double **F, double **G, double **H, int N);

#endif
