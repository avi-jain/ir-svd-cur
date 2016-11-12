import numpy as np
from svd import SVD
import scipy.sparse as sp

start = time.clock()

# Note - numpy's dot function does not have native support for handling sparse matrices
# Hence we make use of scipy's sparse functions
def frobenius_norm(arr):
        """ Frobenius norm for a data matrix        
        Returns:
            frobenius norm: F = ||Sum of squares of all elements in matrix - sum of squares ||
        """                 
        pass
def pinv(W):    
    # Compute Pseudoinverse of a matrix W
    # calculate SVD
    svd = SVD(W)
    X, Z, YT = W._left_svd() 
    XT = X.T
    Y = YT.T

    # Wdiag = W.diagonal()
    # Wdiag = np.where(Wdiag >eps, 1.0/Wdiag, 0.0)
    
    # for i in range(S.shape[0]):
    #     W[i,i] = Wdiag[i]
            
    # if sp.issparse(W):            
    #     U =  Y * ZP  * XT
    # else:    
    #     U = np.dot()

    # return U
    print(YT)
    print(Y)

class CUR:
    """     
    CUR Decomposition. Factorize a data matrix into three matrices s.t.
    F = | data - USV| is minimal. CUR randomly selects cols and rows from
    data for building C and R, respectively.
    The U matrix is the pseudo inverse of W+ which is the SVD of W.
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
        4 (default)            
    """
    
    def __init__(self, data, k=-1, rrank=0, crank=0):
    	#instead of writing similar constructor separately, reuse code
        SVD.__init__(self, data,k=k,rrank=rrank, crank=rrank)

    def sample_probability(self):
        
        if sp.issparse(self.data):
            dsquare = self.data.multiply(self.data)    
        else:
            dsquare = self.data[:,:]**2
        
        # specifying np64 because of future division
        prow = np.array(dsquare.sum(axis=1), np.float64)
        pcol = np.array(dsquare.sum(axis=0), np.float64)
        print(prow)
        print(pcol)

        prow /= prow.sum()
        pcol /= pcol.sum()    
        
        # the unspecified value (no of rows) is inferred from original array for np.reshape
        return (prow,pcol)

    # For picking out rows and cols where n -> no of rows/cols to be picked
    def sample(self, n, prob,type):        
        if(type == "row"):
        	arr = np.empty(n,)
        elif(type == "col")
        for i in range(n):            
            # pick the highest probability row/col,insert column after dividing by root(nP)
            # and then turn that to zero
	        temp_ind = np.argmax(prob)
	        if(type == "row"):
	        	np.append(arr,(self.data[temp_ind,:]/sqrt(n*prob[temp_ind])))
	        elif(type == "col"):
	        	np.append(arr,(self.data[:,temp_ind]/sqrt(n*prob[temp_ind])))
        return arr
    def computeCUR(self):          
        if sp.issparse(self.data):
            self._C = sp.csc_matrix()        
            self._R = sp.csc_matrix()        

            self._U = pinv()
                     
        else:        
            self._C = 
            self._R = 

            self._U = pinv()

data = np.array([
	[2, 5, 3],
	[1, 2, 1],
	[4, 1, 1],
	[3, 5, 2],
	[5, 3, 1],
	[4, 5, 5],
	[2, 4, 2],
	[2, 2, 5],
])
cur = CUR(data)
prow, pcol = cur.sample_probability()
rows = cur.sample(prows,"row")
cols = cur.sample(pcols,"col")

end = time.clock()
print(end - start)

    


        