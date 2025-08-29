#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

matrix transform_PyList_matrix_to_matrix(PyObject *input_list)
{
    matrix empty;
    empty.cols = -1;
    empty.rows = -1;
    empty.mat = NULL;

    // Check if it's actually a list
    if (!PyList_Check(input_list))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a list");
        return empty;
    }

    Py_ssize_t rows = PyList_Size(input_list);
    if (rows == 0)
    {
        PyErr_SetString(PyExc_ValueError, "No rows in the matrix!");
        return empty;
    }

    /* Make sure first row is a valid list and get the amount of columns. */
    PyObject *first_row = PyList_GetItem(input_list, 0);
    if (!PyList_Check(first_row))
    {
        PyErr_SetString(PyExc_TypeError, "Expected a list for a row.");
        return empty;
    }
    Py_ssize_t cols = PyList_Size(first_row);

    /* Make sure the rest of the rows have the same amount of columns. */
    for (Py_ssize_t i = 0; i < rows; i++)
    {
        PyObject *row_list = PyList_GetItem(input_list, i);
        if (!PyList_Check(row_list))
        {
            PyErr_SetString(PyExc_TypeError, "Expected a list for a row.");
            return empty;
        }

        if (cols != PyList_Size(row_list))
        {
            PyErr_SetString(PyExc_ValueError, "All rows must have the same length!");
            return empty;
        }
    }

    /* Create our custom matrix and place the elements. */
    matrix input_matrix = create_matrix(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        PyObject *row_list = PyList_GetItem(input_list, i);
        for (int j = 0; j < cols; j++)
        {
            double val = PyFloat_AsDouble(PyList_GetItem(row_list, j));
            if (PyErr_Occurred())
            {
                PyErr_SetString(PyExc_TypeError, "One of the values in the matrix is not a number.");
                destroy_matrix(&input_matrix);
                return empty;
            }

            input_matrix.mat[i][j] = val;
        }
    }

    /* Return the final matrix. */
    return input_matrix;
}

PyObject *transform_matrix_to_PyList(matrix mat)
{
    PyObject *rows = PyList_New(0);
    for (int i = 0; i < mat.rows; i++)
    {
        PyObject *current_row = PyList_New(0);
        for (int j = 0; j < mat.cols; j++)
        {
            PyList_Append(current_row, PyFloat_FromDouble(mat.mat[i][j]));
        }

        PyList_Append(rows, current_row);
    }
    return rows;
}

static PyObject *py_symnmf(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *py_normalized_matrix;
    PyObject *py_initial_H;
    int k;

    matrix W;
    matrix inital_H;
    matrix output_matrix;

    if (!PyArg_ParseTuple(args, "iOO", &k, &py_normalized_matrix, &py_initial_H))
    {
        return NULL; // Error in parsing arguments
    }
    W = transform_PyList_matrix_to_matrix(py_normalized_matrix);

    inital_H = transform_PyList_matrix_to_matrix(py_initial_H);
    output_matrix = create_matrix(W.rows, k);

    /* Compute m = mean of entries in normalized similarity */
    symnmf_strategy(output_matrix, k, W, inital_H);
    PyObject *output_py_matrix = transform_matrix_to_PyList(output_matrix);

    destroy_matrix(&W);
    destroy_matrix(&inital_H);
    destroy_matrix(&output_matrix);

    return output_py_matrix;
}

static PyObject *py_sim(PyObject *self, PyObject *args)
{
    PyObject *datapoints_matrix;

    if (!PyArg_ParseTuple(args, "O", &datapoints_matrix))
    {
        return NULL;
    }

    matrix input_datapoints = transform_PyList_matrix_to_matrix(datapoints_matrix);
    matrix similarity_matrix = create_matrix(input_datapoints.rows, input_datapoints.rows);
    sim_strategy(similarity_matrix, input_datapoints);

    PyObject *similarity_py_matrix = transform_matrix_to_PyList(similarity_matrix);
    destroy_matrix(&input_datapoints);
    destroy_matrix(&similarity_matrix);

    return similarity_py_matrix;
}

static PyObject *py_ddg(PyObject *self, PyObject *args)
{
    PyObject *datapoints_matrix;

    if (!PyArg_ParseTuple(args, "O", &datapoints_matrix))
    {
        return NULL;
    }

    matrix input_datapoints = transform_PyList_matrix_to_matrix(datapoints_matrix);
    matrix ddg_matrix = create_matrix(input_datapoints.rows, input_datapoints.rows);
    ddg_strategy(ddg_matrix, input_datapoints);

    PyObject *ddg_py_matrix = transform_matrix_to_PyList(ddg_matrix);
    destroy_matrix(&input_datapoints);
    destroy_matrix(&ddg_matrix);

    return ddg_py_matrix;
}

static PyObject *py_norm(PyObject *self, PyObject *args)
{
    PyObject *datapoints_matrix;

    if (!PyArg_ParseTuple(args, "O", &datapoints_matrix))
    {
        return NULL;
    }
    matrix input_datapoints = transform_PyList_matrix_to_matrix(datapoints_matrix);
    matrix norm_matrix = create_matrix(input_datapoints.rows, input_datapoints.rows);
    norm_strategy(norm_matrix, input_datapoints);

    PyObject *norm_py_matrix = transform_matrix_to_PyList(norm_matrix);
    destroy_matrix(&input_datapoints);
    destroy_matrix(&norm_matrix);

    return norm_py_matrix;
}

/* Module methods */
static PyMethodDef SymnmfMethods[] = {
    {"sym", py_sim, METH_VARARGS, "Compute similarity matrix"},
    {"ddg", py_ddg, METH_VARARGS, "Compute diagonal degree matrix"},
    {"norm", py_norm, METH_VARARGS, "Compute normalized similarity matrix"},
    {"symnmf", (PyCFunction)py_symnmf, METH_VARARGS | METH_KEYWORDS, "Run the SymNMF algorithm.\nUsage: symnmf(W, H, k)\n    Where W is the normalized similarity matrix, H is the initial H and k."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    "Python C API wrapper for SymNMF",
    -1,
    SymnmfMethods};

PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    return PyModule_Create(&symnmfmodule);
}
