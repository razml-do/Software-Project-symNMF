#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/**
 * transform_PyList_matrix_to_matrix - Converts a Python list of lists to a C matrix.
 * @input_list: Python list of lists representing a 2D numeric array.
 *
 * Returns: A matrix struct containing the numeric values.
 * In case of any error, prints "An Error Has Occurred" and returns an empty matrix.
 */
matrix transform_PyList_matrix_to_matrix(PyObject *input_list)
{
    matrix empty = {-1, -1, NULL};
    if (!PyList_Check(input_list))
    {
        return empty;
    }

    Py_ssize_t rows = PyList_Size(input_list);
    if (rows == 0)
    {
        return empty;
    }

    PyObject *first_row = PyList_GetItem(input_list, 0);
    if (!PyList_Check(first_row))
    {
        return empty;
    }
    Py_ssize_t cols = PyList_Size(first_row);

    for (Py_ssize_t i = 0; i < rows; i++)
    {
        PyObject *row = PyList_GetItem(input_list, i);
        if (!PyList_Check(row) || PyList_Size(row) != cols)
        {
            return empty;
        }
    }

    matrix mat = create_matrix(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        PyObject *row = PyList_GetItem(input_list, i);
        for (int j = 0; j < cols; j++)
        {
            double val = PyFloat_AsDouble(PyList_GetItem(row, j));
            if (PyErr_Occurred())
            {
                destroy_matrix(&mat);
                return empty;
            }
            mat.mat[i][j] = val;
        }
    }
    return mat;
}

/**
 * transform_matrix_to_PyList - Converts a C matrix to a Python list of lists.
 * @mat: Matrix struct to convert.
 *
 * Returns: A Python list of lists representing the matrix.
 */
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

/**
 * py_symnmf - Python wrapper for the SymNMF algorithm.
 * @self: Python module reference (unused).
 * @args: Tuple containing k, normalized matrix W, and initial H.
 *
 * Returns: Python list of lists representing the output H matrix.
 */
static PyObject *py_symnmf(PyObject *self, PyObject *args)
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

/**
 * Python wrapper for similarity.
 * @self: Python module reference (unused).
 * @args: Python matrix of datapoints.
 *
 * Returns: Python list of lists for the requested matrix type.
 */
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

/**
 * Python wrapper for diagonal degree.
 * @self: Python module reference (unused).
 * @args: Python matrix of datapoints.
 *
 * Returns: Python list of lists for the requested matrix type.
 */
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

/**
 * Python wrapper normalization matrices.
 * @self: Python module reference (unused).
 * @args: Python matrix of datapoints.
 *
 * Returns: Python list of lists for the requested matrix type.
 */
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
    {"symnmf", py_symnmf, METH_VARARGS, "Run the SymNMF algorithm.\nUsage: symnmf(W, H, k)\n    Where W is the normalized similarity matrix, H is the initial H and k."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",                    /* m_name */
    "Python C API wrapper for SymNMF", /* m_doc */
    -1,                                /* m_size */
    SymnmfMethods                      /* m_methods */
};

PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    return PyModule_Create(&symnmfmodule);
}
