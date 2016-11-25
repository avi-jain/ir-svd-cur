# This module implements the CUR functionality

# REQUIRED IMPORTS

import scipy.sparse as sps
import numpy as np



def csr_vappend(a, b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one.
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])


# Pass selected column/rows and row/col indexes
def get_rows_from_col_matrix(z, row_val):
    arr = []
    for i in row_val:
        arr.append(z.getrow(i).toarray()[0])
    return np.array(arr)


def func(z):
    (row,col)=z.shape
    for i in range(row):
        for j in range(col):
            if(z[i][j]!=0):
                z[i][j]=1.0/float(z[i][j])
    return z


def pinv(W):
    # Compute Pseudoinverse of a matrix W
    # calculate np.linalg.svd
    X, Z, YT = np.linalg.svd(W,full_matrices=False)
    Z = np.diag(Z)
    XT = X.T
    Y = YT.T
    ZP = func(Z)

    U = np.dot(Y,np.dot(ZP,XT))

    return U



class CUR:
    """
    CUR Decomposition. Factorize a data matrix into three matrices s.t.
    F = | data - USV| is minimal. CUR randomly selects cols and rows from
    data for building C and R, respectively.
    The U matrix is the pseudo inverse of W+ which is the np.linalg.svd of W.
    W is the "intersection" of C and R
    Parameters
    ----------
    data : array_like [data_dimension x num_samples]
        the input data
    rrank: int, optional
        Number of rows to sample from data.
        4 (default)
    crank: int, optional
        Number of columns to sample from data.
        2 (default)
    """

    def __init__(self, data):
        self.rows, self.cols = data[0], data[1]
        self.data = sps.csr_matrix((data[2], (self.rows, self.cols)))


    def sample_probability(self):
        dsquare = self.data.multiply(self.data)

        # specifying np64 because of future division
        prow = np.array(dsquare.sum(axis=1), np.float64)
        pcol = np.array(dsquare.sum(axis=0), np.float64)
        # print(prow)
        # print(pcol)

        prow /= prow.sum()
        pcol /= pcol.sum()

        # the unspecified value (no of rows) is inferred from original array
        # for np.reshape
        return (prow, pcol)


    # For picking out rows and cols where n -> no of rows/cols to be picked
    def sample(self, n, prob, typ):
        probcum = np.cumsum(prob.flatten())

        ind = []

        if typ == "row":
            matrices = sps.csr_matrix((0, self.data.shape[1]))
        else:
            matrices = sps.csr_matrix((0, self.data.shape[0]))

        for i in range(n):
            # pick the row/col,insert column after dividing by root(nP)
            r = np.random.rand()
            for j in range(len(probcum)):
                if(probcum[j] >= r):
                    # print(temp_ind)
                    # push into another array for intersection
                    ind.append(j)
                    # add a row/column to the list of matrices we're building
                    if typ == "row":
                        csr_vappend(matrices, self.data.getrow(j))
                    else:
                        col = self.data.getcol(j).transpose().tocsr()
                        csr_vappend(matrices, col)
                    break

        if typ == "col":
            matrices = matrices.transpose().tocsr()

        # Returns indices of selected rows/cols
        return (ind, matrices)
