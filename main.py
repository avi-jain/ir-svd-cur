# THE MAIN DRIVER PROGRAM

# REQUIRED LIBRARRIES

import sys

import scipy.sparse as sps

import svd
import cur
import numpy as np
import data_loader
import time


# TEST
sample = [
    [(0,2.), (1,5.), (2,3.)],
    [(0,1.), (1,2.), (2,1.)],
    [(0,4.), (1,1.), (2,1.)],
    [(0,3.), (1,5.), (2,2.)],
    [(0,5.), (1,3.), (2,1.)],
    [(0,4.), (1,5.), (2,5.)],
    [(0,2.), (1,4.), (2,2.)],
    [(0,2.), (1,2.), (2,5.)],
]

def process_data(data):
    rows = []
    cols = []
    output = []
    for row in range(len(data)):
        for (col, val) in data[row]:
            rows.append(row)
            cols.append(col)
            output.append(val)
    return (np.array(rows), np.array(cols), np.array(output))



def main():

    k = 4

    if len(sys.argv) < 2:
        print("ERROR : You need to specify the input file")
        quit()

    data_file = sys.argv[1]

    # Get sparse matrix of data
    data = process_data(data_loader.get_data(sys.argv[1]))
    a = svd.SVD(data)
    b = cur.CUR(data)

    start = time.clock()
    U,S,V = a.left_svd(k=k)
    stop = time.clock()

    reconst = np.dot(U, np.dot(S, V))
    print("U : ")
    print(U)
    print()
    print("S : ")
    print(S)
    print()
    print("V : ")
    print(V)
    print()
    print("Reconstructed SVD : ")
    print(reconst)
    print()
    print()

    print("TIme taken : ", stop - start)


    start = time.clock()

    prow, pcol = b.sample_probability()
    row_ind, rows = b.sample(4 * k, prow, "row")
    col_ind, cols = b.sample(4 * k, pcol, "col")

    print("C:")
    print(cols)
    print()

    #cols = cols.tocsr()
    #print(type(cols))

    W = cur.get_rows_from_col_matrix(cols, row_ind).T
    #print(W)

    print()

#    print(W)
    U = cur.pinv(W)

    print("U:")
    print(U)

    print("R:")
    print(rows)
    print()

    print("CUR:")
    CUR = np.dot(cols.todense(), np.dot(U, rows.todense()))

    stop = time.clock()

    print(CUR)

    print("TIme taken : ", stop - start)

if __name__ == "__main__":
    main()
