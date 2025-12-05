import numpy as np
import findiff as fd
import matplotlib.pyplot as plt
from . import Frame_Builder as fb
import os
import pickle


def eigen_value_decomp(A):
    try:
        eigenvalues, eigenvectors= np.linalg.eig(A)
        eigval_hat= np.linalg.inv(eigenvectors) @ A @ eigenvectors
        
        diagonal_MSE= np.linalg.norm( np.diag(np.diag(eigval_hat)) )  / np.linalg.norm( eigval_hat )           
        is_diag=False
        if diagonal_MSE>0.999: # why this value?
            is_diag=True
    except Exception:
        is_diag=False
        eigenvectors=None
        eigenvalues=None
    return eigenvalues, eigenvectors, is_diag

def get_effective_rank(A,tol=1e-12):
    """Compute the effective rank of a matrix."""
    U, s, Vh = np.linalg.svd(A)
    normalized_s = s / np.sum(s)
    normalized_s = normalized_s[normalized_s > tol]  # Filter out small singular values
    entropy = -np.sum(normalized_s * np.log(normalized_s))
    return np.exp(entropy)

# def get_effective_rank(A,tol=1e-12):
#     """Compute the effective rank of a matrix."""
#     U, s, Vh = np.linalg.svd(A)
#     normalized_s = s**2 / np.sum(s**2)
#     Z= np.argwhere(  np.cumsum(normalized_s) >1-1e-9  )
#     return 1+Z[0][0]


class SSM:
    """
    Attributes: 
    Fobj, Frame object containing frame, dual, and derivative
    N, number of coefficients
    L, length of frame
    fname, name of the function used to generate frame (eg, legendre, fourier)
    meas, measure (eg, scaled, translated)
    """
    def __init__(self, params, Fobjgiven=None):
        # N= params.get("N",50) # worried this will overwrite or mismatch?
        # L= params.get("L",10000)
        self.fname= params.get("fname", "safari custom frame")
        self.meas= params.get("meas", 'scaled')
        save_path=params.get("sav_path",None)

        # 1) if the user has provided a frame, use it.
        if Fobjgiven is not None: 
            print("Using provided frame")
            self.Fobj = Fobjgiven
            if self.fname is None:
                self.fname = 'custom'
    
        # Otherwise, make it from one of the built-in frames provided.
        else:
            self.Fobj = fb.Fobj(self.fname, params)

        self.N = self.Fobj.F.shape[0]
        self.L = self.Fobj.F.shape[1]
        self.completeFrame() # compute dual and derivative if not already provided 
        # (may be provided by user explicitly, or may be an analytical version from our frame builder.)
        
        # 2) check whether there is a known closed-form solution for A, B
        if ((self.fname=='legendre' and self.meas=='scaled') or (self.fname=='legendre' and self.meas=='translated') 
            or (self.fname=='fourier' and self.meas=='translated')):
            self.hippo()
        else:
            self.safari()
        
        #Takes A, and diagonalize it. self.is_diag indicates whether the ssm can be *diagonalized*. (nothing about stability!!)
        # If self.id_diag is calculated to be False, then you should not use the diagonal solution.
        self.eig_val, self.eig_vec, self.B_diag, self.is_diag = self.diagonalize()
           
        # Save if path is provided
        if save_path is not None:
            self.save(save_path)
            
    def save(self, path):
        """Save the current SSM object to the given path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"SSM object saved to {path}")

    @staticmethod
    def load(path):
        """Load an SSM object from the given path with exception handling."""
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            raise FileNotFoundError(f"No such file: '{path}'")
        except pickle.UnpicklingError:
            raise ValueError(f"The file at '{path}' is not a valid SSM pickle.")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred while loading: {e}")      
            

    def completeFrame(self):
        # if a dual of the frame has been provided, use it.  Else, compute it.
        if self.Fobj.D is None:
            print("Computing dual frame")
            self.Fobj.D = np.linalg.pinv(self.Fobj.F).T
        # if the derivative of the frame has been provided, use it.  Else, compute it.
        if self.Fobj.dF is None:
            print("Computing derivative of frame")
            self.Fobj.dF = self.fdFrame(self.Fobj.F)

    def hippo(self):

        A = np.zeros((self.N,self.N))
        B = np.zeros((self.N,1))

        if self.fname=='legendre' and self.meas=='scaled':
            print('Generating HiPPO-legS')
            for n in range(self.N):
                B[n] = np.sqrt(2*n+1)
                for k in range(n+1):
                    if n == k:
                        A[n,k] = n+1
                    else:
                        A[n,k] = np.sqrt(2*n+1)*np.sqrt(2*k+1) 

        elif self.fname=='legendre' and self.meas=='translated':
            print('Generating HiPPO-legT')
            for n in range(self.N):
                B[n] = np.sqrt(2*n+1)
                for k in range(self.N):
                    if n <= k:
                        A[n,k] = np.sqrt(2*n+1)*np.sqrt(2*k+1)*((-1)**(n-k))
                    else:
                        A[n,k] = np.sqrt(2*n+1)*np.sqrt(2*k+1) 

        elif self.fname=='fourier' and self.meas=='translated':
            print('Generating HiPPO-fouT')
            B[0]=1
            for n in range(self.N):
                if n%2==1:
                    B[n]=np.sqrt(2)
                for k in range(self.N):
                    if n == 0 and k == 0:
                        A[n,k] = 1
                    if n%2 == 1: 
                        if k == 0: # n odd, k=0
                            A[n,k] = np.sqrt(2) 
                        elif k%2 == 1: # n,k both odd
                            A[n,k] = 2
                        elif k-n == 1: # n odd, k-n = 1
                            A[n,k] = -np.pi*(n+1)
                    elif k%2 == 1: 
                        if n == 0: # k odd, n=0
                            A[n,k] = np.sqrt(2)
                        elif n-k == 1: # k odd, n-k = 1
                            A[n,k] = np.pi*(k+1)
        self.A = A
        self.B = B
    
    def safari(self):

        L = self.Fobj.F.shape[1]

        if self.meas == 'scaled':
            dF = self.Fobj.dF 
            Theta = (np.arange(L)/L)*dF
            t = np.eye(self.N)

        elif self.meas == 'translated':
            Theta = self.Fobj.dF
            t = (self.Fobj.F[:,0][:,None] @ self.Fobj.D[:,0][None,:])*L

        A = t + (Theta @ self.Fobj.D.T)
        B = self.Fobj.F[:,L-1]
    
        self.A = A
        self.B = B[:,None]
            
    def fdFrame(self, F):
    # input:    F, the frame
    # output:   dF, first derivative of the frame calculated by finite difference method
        dx = 1/self.L
        dF_dx = fd.FinDiff(1,dx)
        dF = dF_dx(F)
        return dF
    
    def diagonalize(self):
        eigenvalues, eigenvectors,is_diag=eigen_value_decomp(self.A)
        self.erank = get_effective_rank(self.A)
        if is_diag:
            B_diag= np.linalg.inv(eigenvectors) @ self.B
            # eff_rank = int(np.floor(self.erank)) # effective rank is real-valued, need integer here
            #This part of the code numerically identifies the effective number of eigenvalues that contribute in diagonalization
            if self.meas=='scaled':
                eff_rank= 1+len(np.argwhere( np.abs(eigenvalues)>1.000001 )) 
            elif self.meas=='translated':
                eff_rank= 1+len(np.argwhere( np.abs(eigenvalues)>0.0000001 )) 
            eff_rank= min(eff_rank, self.A.shape[0])
            print(self.fname, self.meas,", diagonalizable, effective rank:", eff_rank )
            Translation= self.Fobj.F  @ self.Fobj.D.T
            eigenvalues= eigenvalues[0: eff_rank]
            eigenvectors=Translation @ eigenvectors[ :, 0:eff_rank ]
            B_diag= B_diag[0:eff_rank]
            
            return eigenvalues.squeeze(), eigenvectors, B_diag.squeeze(), is_diag
        else:
            print(self.fname,", non-diagonalizable, effective rank:", self.A.shape[0] )
            return None, None, None, is_diag


