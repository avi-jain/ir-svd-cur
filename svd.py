# This module implements the SVD functionality

# REQUIRED IMPORTS

import scipy.sparse as sps
from scipy.sparse.linalg import eigsh
import numpy as np


class SVD:
    def __init__(self, data):
        self.rows, self.cols = data[0], data[1]
        self.data = sps.csr_matrix((data[2], (self.rows, self.cols)))

    # The left singular value decomposition (compute U, then use that to get V)
    def left_svd(self, k=-1):
        # Calculate (A . A_transpose)
        AAT = self.data.dot(self.data.transpose())

        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigsh(AAT)

        # Take only the 'k' largest eigenvalues and corresponding eigenvectors
        if k <= 0:
            index = np.argsort(eigenvalues)[::-1]
        else:
            index = np.argsort(eigenvalues)[-k:][::-1]
        eigenvalues = eigenvalues[index]
        eigenvectors = eigenvectors[:,index]

        # U is basically the surviving eigenvectors
        self.U = eigenvectors[:]

        # Compute the sigma matrix
        # Take square root of non zero eigenvalues and populate the diagonal
        # with them, putting the largest in (0,0) and the smallest in (k,k)
        self.S = np.diag(np.sqrt(abs(eigenvalues)))

        # and inverse of it
        S_inv = np.diag(np.sqrt(abs(eigenvalues))**-1)

        # Compute V from it
        self.V = np.dot(S_inv, np.dot(self.U[:,:].T, self.data.todense()))
        return (self.U, self.S, self.V)
