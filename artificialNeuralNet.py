# Functions that would be used to create the topography of the neuraal net
# The neurons and weights are taken as matrices
# Neurons are 1D arrays or lists of the dimension 1 X col

import numpy as np
import pandas as pa


def collectData (sheet_name):
    """ Returns an array (numpy) of the input data from the excel sheet """

    Input = pa.read_excel(sheet_name) # I is the input data matrix, data has to be extracted one by one from the columns
    Input = np.array(Input)

    return Input


def getError(Input, row, Output):
    """ Returns the error value of the network """

    col = int(np.shape(Input)[1]) - 1
    error = 0.5* ( (Input[row][col] - Output)**2 ) # MSE

    return error

def inputNeurons (Input, row):
    """ Returns an input matrix based on the data matrix with data set present in column 'column' """

    n_row = int(np.shape(Input)[0])
    n_col = int(np.shape(Input)[1])

    I = [0]*(n_col - 1)

    for c in range(n_col-1):
        I[c] = Input[row][c]

    return I


def transposeMat (M):
    """ Returns the transpose of matrix M, used for neurons in the next layer """
    # Not used in the current program

    M = np.array(M)
    row = int(np.shape(M)[0])
    col = int(np.shape(M)[1])

    M_trans = np.zeros((col, row))

    for r in range(row):
        for c in range(col):
            M_trans[c][r] = M[r][c].copy()

    M_trans = M_trans.tolist()

    return M_trans


def getNextLayer (n_out_mat, w_mat):
    """ Gets the next layer from output matrix of neurons and weights """
    # The layer is in the form of 1 X col array/matrix

    N = np.array(n_out_mat)
    W = np.array(w_mat)

    if (W.ndim == 1):
        col_wt = int(np.shape(W)[0])
    else:
        col_wt = int(np.shape(W)[1])
    col_n = int(np.shape(N)[0])

    M_mult = np.zeros((1, col_wt)) # Only designed for neurons i.e. matrices of size 1 X col

    for c_w in range(col_wt):
        for c_n in range(col_n):
            if (W.ndim == 1):
                M_mult[0][c_w] = M_mult[0][c_w] + ( N[c_n] * W[c_w] )
            else:
                M_mult[0][c_w] = M_mult[0][c_w] + ( N[c_n] * W[c_n][c_w] ) # r X c Type matrix multiplication

    M_mult = M_mult.tolist()

    return M_mult[0]