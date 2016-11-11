from numpy.linalg import svd,eigh
import numpy as np
bookRatings = [
	[2, 5, 3],
	[1, 2, 1],
	[4, 1, 1],
	[3, 5, 2],
	[5, 3, 1],
	[4, 5, 5],
	[2, 4, 2],
	[2, 2, 5],
]

U, singularValues, V = svd(bookRatings)
print(U)
print(V)

# Implementing the code
# Instead of python arrays, we use numpy arrays
# numpy.ndarray.shape() returns a tuple of (row,col)
class SVD:
	def __init__(self, data, k=-1, rowrank=0, colrank=0):
		self.data = data
		(self._rows, self._cols) = self.data.shape
		if rowrank > 0:
			self._rowrank = rowrank
		else:
			self._rowrank = self._rows
			
		if colrank > 0:            
			self._colrank = colrank
		else:
			self._colrank = self._cols
		
		# set the overall rank
		self._k = k

	# The left singular matrix U
	def _left_svd(self):
			# [:,:] -> from start to end
			# np.dot performs the dot product
			# numpy.matrix.T - Returns the transpose of the matrix.        
			AAT = np.dot(self.data[:,:], self.data[:,:].T)
			
			# Return the eigenvalues and eigenvectors
			values, u_vectors = eigh(AAT) 

			# Can remove eigenvalues that are very low
							
			# sort eigenvectors according to largest eigenvalue
			sort_perm = values.argsort()
			u_vectors = u_vectors[:, sort_perm]

			# compute the Gram-Schimdt orthonormalization - basically normalize magnitude
			print(u_vectors)
			# compute Sigma matrix- 
			# take the square roots of the non-zero Eigenvalues 
			# and populate the diagonal with them, 
			# putting the largest in Ʃ11, the next largest in Ʃ22 and so on 
			# until the smallest value ends up in Ʃmm
			# self.U = u_vectors[]

			# self.S = np.diag(np.sqrt(values))
			
			# # and the inverse of it
			# S_inv = np.diag(np.sqrt(values)**-1)
					
			# # compute V from it
			# self.V = np.dot(S_inv, np.dot(self.U[:,:].T, self.data[:,:]))

	# The right singular matrix V
	def _right_svd(self):
			ATA = np.dot(self.data[:,:].T, self.data[:,:])
			values, v_vectors = eigh(ATA)    
			
			# sort eigenvectors according to largest eigenvalue
			sort_perm = values.argsort()
			v_vectors = v_vectors[:, sort_perm]
			print(v_vectors)
			# compute the Gram-Schimdt orthonormalization - basically normalize magnitude

			# Add values to Sigma
			# self.S= np.diag(np.sqrt(values))
			
			# # and the inverse of it
			# S_inv = np.diag(1.0/np.sqrt(values))    
						
			# Vtmp = v_vectors[]

			# self.U = np.dot(np.dot(self.data[:,:], Vtmp), S_inv)                    
			# self.V = Vtmp.T
data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
svd = SVD(data)
svd._left_svd()
svd._right_svd()