#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "symnmf.h"

/* Helper: convert a C double** matrix to a NumPy array */
static PyObject *matrix_to_numpy(double **mat, int rows, int cols)
{
    npy_intp dims[2] = {rows, cols};
    PyObject *array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!array)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate NumPy array");
        return NULL;
    }

    double *data = (double *)PyArray_DATA((PyArrayObject *)array);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            data[i * cols + j] = mat[i][j];

    return array;
}

/* Wrapper for similarity matrix */
static PyObject *py_sym(PyObject *self, PyObject *args)
{
    const char *data_csv;
    if (!PyArg_ParseTuple(args, "s", &data_csv))
        return NULL;

    int N, d;
    double **data = read_csv_file(data_csv, &N, &d);
    if (!data)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to read CSV");
        return NULL;
    }

    double **A = similarity_matrix(data, d, N);
    if (!A)
    {
        free_all(data, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, N);
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute similarity matrix");
        return NULL;
    }

    PyObject *result = matrix_to_numpy(A, N, N);
    free_all(data, A, NULL, NULL, NULL, NULL, NULL, NULL, NULL, N);
    return result;
}

/* Wrapper for diagonal degree matrix */
static PyObject *py_ddg(PyObject *self, PyObject *args)
{
    const char *data_csv;
    if (!PyArg_ParseTuple(args, "s", &data_csv))
        return NULL;

    int N, d;
    double **data = read_csv_file(data_csv, &N, &d);
    if (!data)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to read CSV");
        return NULL;
    }

    double **A = similarity_matrix(data, d, N);
    double **D = diagonal_degree_matrix(A, N);
    PyObject *result = matrix_to_numpy(D, N, N);
    free_all(data, A, NULL, NULL, D, NULL, NULL, NULL, NULL, N);
    return result;
}

/* Wrapper for normalized similarity matrix */
static PyObject *py_norm(PyObject *self, PyObject *args)
{
    const char *data_csv;
    if (!PyArg_ParseTuple(args, "s", &data_csv))
        return NULL;

    int N, d;
    double **data = read_csv_file(data_csv, &N, &d);
    if (!data)
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to read CSV");
        return NULL;
    }

    double **A = similarity_matrix(data, d, N);
    double **D = diagonal_degree_matrix(A, N);
    double **D_inv_sqrt = diagonal_inverse_square_root(D, N);
    double **tmp = mat_mul(D_inv_sqrt, N, N, A, N, N);
    double **norm = mat_mul(tmp, N, N, D_inv_sqrt, N, N);

    PyObject *result = matrix_to_numpy(norm, N, N);
    free_all(data, A, D_inv_sqrt, tmp, D, norm, NULL, NULL, NULL, N);
    return result;
}

/* Wrapper for full SymNMF */
static PyObject *py_symnmf(PyObject *self, PyObject *args, PyObject *kwargs)
{
    const char *data_csv;
    const char *H_csv = NULL;
    const char *W_csv = NULL;
    int k = 0;

    static char *kwlist[] = {"data_csv", "H_csv", "W_csv", "k", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|ssi", kwlist,
                                     &data_csv, &H_csv, &W_csv, &k))
        return NULL;

    if (!H_csv || !W_csv || k <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "H_csv, W_csv, and k must be provided");
        return NULL;
    }

    int N, d;
    double **data = read_csv_file(data_csv, &N, &d);
    int H_rows, H_cols, W_rows, W_cols;
    double **H = read_csv_file(H_csv, &H_rows, &H_cols);
    double **W = read_csv_file(W_csv, &W_rows, &W_cols);

    if (!data || !H || !W || H_rows != N || H_cols != k || W_rows != N || W_cols != N)
    {
        free_all(data, H, W, NULL, NULL, NULL, NULL, NULL, NULL, N);
        PyErr_SetString(PyExc_ValueError, "Invalid input dimensions");
        return NULL;
    }

    const int max_iter = 300;
    const double beta = 0.5, epsilon = 1e-4;
    double **WH = NULL, **H_transpose = NULL, **HHT = NULL, **HHTH = NULL;
    double **ratio_mat = NULL, **beta_mul_and_addition = NULL, **frob_sub = NULL;
    double **H_new = NULL;

    for (int iter = 0; iter < max_iter; iter++)
    {
        WH = mat_mul(W, N, N, H, N, k);
        H_transpose = mat_transpose(H, N, k);
        HHT = mat_mul(H, N, k, H_transpose, k, N);
        HHTH = mat_mul(HHT, N, N, H, N, k);

        /* clamp small values to avoid division by zero */
        for (int i = 0; i < N; i++)
            for (int j = 0; j < k; j++)
                if (HHTH[i][j] < 1e-12)
                    HHTH[i][j] = 1e-12;

        ratio_mat = mat_div(WH, HHTH, N, k);
        beta_mul_and_addition = mat_const_mul_and_add(ratio_mat, N, k, beta, 1 - beta);
        H_new = mat_mul_indices(H, beta_mul_and_addition, N, k);

        frob_sub = mat_sub(H_new, H, N, k);
        double squared_frob = frobenius_norm_squared(frob_sub, N, k);

        iterative_free(WH, H_transpose, HHT, HHTH, ratio_mat,
                       beta_mul_and_addition, frob_sub, N, k);

        double **oldH = H;
        H = H_new;
        H_new = NULL;
        free_matrix(&oldH, N);

        if (squared_frob < epsilon)
            break;
    }

    PyObject *result = matrix_to_numpy(H, N, k);
    free_all(data, H, W, NULL, NULL, NULL, NULL, NULL, NULL, N);
    return result;
}

/* Module methods */
static PyMethodDef SymnmfMethods[] = {
    {"sym", py_sym, METH_VARARGS, "Compute similarity matrix"},
    {"ddg", py_ddg, METH_VARARGS, "Compute diagonal degree matrix"},
    {"norm", py_norm, METH_VARARGS, "Compute normalized similarity matrix"},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS | METH_KEYWORDS, "Run SymNMF algorithm"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    "Python C API wrapper for SymNMF",
    -1,
    SymnmfMethods};

PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    import_array();
    return PyModule_Create(&symnmfmodule);
}
