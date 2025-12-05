import numpy as np
import math
import findiff as fd
from . import Utils


class Fobj:
    """
    Frame object contains the N x L frame (or basis) F, 
    the dual frame D, and the derivative of the frame dF.
    """

    def __init__(self, fname, params):
        self.generateFrame(fname, params)
    
    def make_frame(self, frame, rcond=0.01):
        N,L=frame.shape
        dF = np.empty((N, L), dtype=float)
        d_dt=fd.FinDiff(0,1/L)   #prepares  the finite difference module to fidn the dervative of Frame
        for i in range(N):
            dF[i,:]= d_dt(frame[i, :])
            
        self.F= frame
        self.dF= dF
        self.D= np.linalg.pinv(frame, rcond=rcond).T 

    def generateFrame(self, fname, params):
    # input:    N, number of coefficients
    #           L, length of basis of signal (only matters for numerical accuracy)
    #           type
    # output:   F, an NxL frame (or basis) on interval
    #           D, the dual frame (only if analytical solution available)
    #           dF, the derivative of the frame (only if analytical solution available)
    
        N = params.get("N", 100)    #if N is provided, use it, otherwise use the default value of 100
        L = params.get("L", 5000)   #similar to the above
        wavelet_name=params.get("wavelet_name", 'db8') # This defaults the wavelet kind to db8
        wavelet_scale_min=params.get("wavelet_scale_min", -2) # This defaults the wavelet kind to db8
        wavelet_scale_max=params.get("wavelet_scale_max", 0) # This defaults the wavelet kind to db8
        wavelet_shift=params.get("wavelet_shift", 0.01) # This defaults the wavelet kind to db8
        

        range_min=params.get("range_min", 0.0)
        range_max=params.get("range_max", 1.0)
        rcond=params.get("rcond", 0.01) 

        if fname=='legendre':
            print('Generating Legendre basis of size ',N,'x',L)
            F = np.zeros([N,L])
            norm = np.sqrt(2*np.arange(N)+1)[:,None] # normalization vector for scaled legendre
            # np.polynomial.legendre generates a *single polynomial* with coefficients for degree
            # Eg, passing (1,2,3) gives you P = 1P0 + 2P1 + 3P2.
            # We want to keep each one separate for the basis, so we'll generate them individually.
            for i in range(N):
                # To get a Legendre polynomial of order 2, we can pass (0,0,1,0), to get order 3 we'd pass (0,0,0,1)...
                # I will combine this with the scaling as (0,0,c2,0), (0,0,0,c3), etc.
                coef = np.zeros(N,)
                coef[i] = norm[i]
                # Legendre polynomial object with coefficients (0,0,...c_i,..0)
                # evaluated on [-1,1] and mapped to [0,1]
                p = np.polynomial.legendre.Legendre(coef,[0,1],[-1,1]) 
                # p is a polynomial object and we want a vector
                (x,F[i,:]) = p.linspace(L) # evaluate over domain [0,1] at L points 
            D = F / L # orthogonal basis, so the dual (inverse) is itself.
            # we've implicitly scaled F by L though, so we need to divide by L in the inverse.
        
        elif fname=='fourier':
            print('Generating fourier basis of size ',N,'x',L)
            lvl = N//2
            F= np.zeros((1+2*lvl,L))
            D= np.zeros((1+2*lvl,L))
            dF=np.zeros((1+2*lvl,L))
            x= np.arange(L)/L
            F[0:L]= 1
            D[0:L]= 1
            for i in range(lvl):
                F[2*i+1,:]= 2**0.5 * np.cos( 2*np.pi * (i+1) * x )
                F[2*i+2,:]= 2**0.5 * np.sin( 2*np.pi * (i+1) * x )
                # derivative is available analytically, so we will produce it here.
                dF[2*i+1,:]= -2**0.5 * 2*np.pi*(i+1) * np.sin( 2*np.pi * (i+1) * x ) 
                dF[2*i+2,:]= 2**0.5 * 2*np.pi*(i+1) * np.cos( 2*np.pi * (i+1) * x ) 
            D = F / L # orthogonal basis, so the dual (inverse) is itself.

        elif fname=='chebyshev':
            x = np.linspace(-1, 1, L)
            F = np.empty((N, L), dtype=float)
            for i in range(N):
                Ti = np.polynomial.chebyshev.Chebyshev.basis(i)     # T_i
                F[i, :] = Ti(x)
            D = np.linalg.pinv(F, rcond=rcond).T 

        elif fname=='laguerre':
            dmax = N*10 # heuristically -- need 10x domain to see convergence
            x = np.linspace(0, dmax, L)
            F = np.empty((N, L), dtype=float)
            for i in range(N):
                Li = np.polynomial.laguerre.Laguerre.basis(i,domain=[0,dmax],window=[0,dmax])     # L_i
                tmp = Li(x)
                F[i, :] = tmp * np.exp(-x/2)  # orthogonality imposed
            D = np.linalg.pinv(F, rcond=rcond).T

        elif fname=='bernstein':
            print('Generating Bernstein basis of size ',N,'x',L)
            x = np.linspace(0, 1, L)
            F = np.empty((N, L), dtype=float)
            n = N - 1  # Bernstein polynomials are degree n with n+1 basis functions
            for i in range(N):
                coeff = math.comb(n, i)
                F[i, :] = coeff * (x**i) * ((1 - x)**(n - i))
            D = np.linalg.pinv(F, rcond=rcond).T

        elif fname=='wavelet':
            L0= len( Utils.mother_reader(wavelet_name,level=1) )
            Level=1+ int( np.log(L/L0)/np.log(2)  )
            print("L:",L, "current level l:",  len( Utils.mother_reader(wavelet_name,level=Level) )  )
            #fetching mother and father wavelets
            mother=Utils.mother_reader(wavelet_name,level=Level)
            father=Utils.father_reader(wavelet_name,level=Level)
            
            #finding the derivative of father and mother wavelets
            d_dt=fd.FinDiff(0,1/len(mother))

            mother_dot= d_dt(mother)
            father_dot= d_dt(father)
            
            # #resampling all of them tothe same length
            # mother= Utils.resample( mother ,L)
            # father= Utils.resample( father ,L)
            # mother_dot= Utils.resample( mother_dot ,L)
            # father_dot= Utils.resample( father_dot ,L)
            
            
            F=     Utils.DWT_filters(mother    ,father    ,wavelet_scale_min, wavelet_scale_max,wavelet_shift)
            # dF=Utils.FrameDerivative(mother_dot,father_dot,wavelet_scale_min, wavelet_scale_max,wavelet_shift)            
            
            
            # #############################################################################
            # #This part is optional. What it does is it removes those filters with minimal intersection to x in[0,1]
            # #############################################################################
            Energies=np.zeros(F.shape[0])
            for i in range(F.shape[0]):
                Energies[i]= np.linalg.norm(F[i,:])
            Energy_max= np.max(Energies)
            keep=[]
            for i in range(F.shape[0]):
                if(Energies[i]>0.005*Energy_max):
                    keep.append(i)
            F=F[keep,:]
            # dF=dF[keep,:]
            
            # ############################end of the optional part#########################
            
            D= np.linalg.pinv(F, rcond=rcond).T
            
        # elif fname=='manual': # MW: this isn't necessary?
            # F=0
            # D=0
            # dF=0
            
        else: 
            raise ValueError("Unknown frame type for built-in frame generator.")
    
            
        
        self.F = F
        try:
            dF
            self.dF = dF
        except:
            self.dF = None
        try:
            D
            self.D = D
        except:
            self.D = None
        