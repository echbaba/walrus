

import numpy as np
import pywt
import scipy



def resample(x, L2):
    L=len(x)
    if np.iscomplexobj(x):
        Z= np.zeros(3*L) * (1j)
    else:
        Z= np.zeros(3*L)
    Z[0:L]=np.flip(x)
    Z[L:2*L]=x
    Z[2*L:3*L]=np.flip(x)
    Y= scipy.signal.resample(Z, 3*L2)
    return Y[L2:2*L2]
    



def resample_and_shift(X,s,shift):
    #stretches by a factor of S, then shift it to the right by the value of shift. The rest would get padded with zero.
    L=len(X)
    if(s!=1):
        X2=resample(X,int(L*s))
    else:
        X2=1*X
    L2=len(X2)
    X3=0*X
    if(s<1): #squeeze
        if(shift>0):
            if(L2+shift<L):
                X3[ shift: L2+shift  ]=X2
            elif(shift<L):
                X3[ shift: L  ]=X2[0: L-shift]
                
        elif(L2>-shift):
            X3[0:L2+shift]= X2[-shift:L]
    elif(s>=1): #stretch
        if(shift<0):
            if(-shift+L<L2):
                X3=X2[-shift: L-shift  ]
            elif((L2+shift>0)):
                X3[0: L2+shift]=X2[-shift: L2  ]
        elif(L>shift):
            X3[shift: L]=X2[0: L-shift] 

    X3=X3/np.sqrt(s)
    return X3



def mother_reader(wavelet_name,level):
    w=pywt.DiscreteContinuousWavelet(wavelet_name)
    if(len(w.wavefun(level))==3):
        (phi, psi, x) = w.wavefun(level)
    elif(len(w.wavefun(level))==2  ):
        (psi,x) = w.wavefun(level)
    else:
        (phi_d, psi, phi_r, psi_r, x) = w.wavefun(level=5)
    return np.array(psi)

def father_reader(wavelet_name,level):
    w=pywt.DiscreteContinuousWavelet(wavelet_name)
    if(len(w.wavefun(level))==3):
        (phi, psi, x) = w.wavefun(level)
    elif(len(w.wavefun(level))==2  ):
        (psi,x) = w.wavefun(level)
        return np.array(psi)
    else:
        (phi_d, psi, phi_r, psi_r, x) = w.wavefun(level=5)
    return np.array(phi)






def DWT_filters(mother,father,s_min,s_max,eta):
    scale_factor=2.0
    L=len(mother)
    F=[]
    
    kmin=int( np.ceil(-1/eta) )
    kmax=int( np.floor( ( scale_factor**(-s_max)   )/eta) )
    kmax= max( 0 , kmax)
    shift= eta * L * scale_factor**(s_max)
    for k in range( kmin , kmax+1):
        F.append( np.conjugate( resample_and_shift(father,scale_factor**(s_max) ,int( k*shift )  ) ) )
    
    for S_inq in range(s_max,s_min-1,-1):
        kmin=int( np.ceil(-1/eta) )
        kmax=int( np.floor( ( scale_factor**(-S_inq)   )/eta) )
        kmax= max( 0 , kmax)
        shift= eta * L * scale_factor**(S_inq)
        #print(S_inq,": ",kmax-kmin )
        for k in range( kmin , kmax+1):
            F.append( np.conjugate( resample_and_shift(mother,scale_factor**(S_inq) ,int( k*shift )  ) ) )
    return np.asarray(F)

# def FrameDerivative(mother_dot,father_dot,s_min,s_max,eta):
#     scale_factor=2.0
#     L=len(mother_dot)
#     F=[]
    
#     kmin=int( np.ceil(-1/eta) )
#     kmax=int( np.floor( ( scale_factor**(-s_max)   )/eta) )
#     kmax= max( 0 , kmax)
#     shift= eta * L * scale_factor**(s_max)
#     for k in range( kmin , kmax+1):
#         F.append( scale_factor**(-s_max)*np.conjugate( resample_and_shift(father_dot,scale_factor**(s_max) ,int( k*shift )  ) ) )

    
#     for S_inq in range(s_max,s_min-1,-1):
#         kmin=int( np.ceil(-1/eta) )
#         kmax=int( np.floor( ( scale_factor**(-S_inq)   )/eta) )
#         kmax= max( 0 , kmax)
#         shift= eta * L * scale_factor**(S_inq)
#         for k in range( kmin , kmax+1):
#             F.append( scale_factor**(-S_inq)*np.conjugate( resample_and_shift(mother_dot,scale_factor**(S_inq) ,int( k*shift )  ) ) )
        
#     return np.asarray(F)





def resample(x, L2):
    L=len(x)
    if np.iscomplexobj(x):
        Z= np.zeros(3*L) * (1j)
    else:
        Z= np.zeros(3*L)
    Z[0:L]=np.flip(x)
    Z[L:2*L]=x
    Z[2*L:3*L]=np.flip(x)
    Y= scipy.signal.resample(Z, 3*L2)
    return Y[L2:2*L2]
    

class peak_detection:
    
    @staticmethod
    def peakfinder(signal2,window_size, threshold):
        signal=1.0*signal2
        signal[signal<0.003]=0
        # window_size=20
        signal=  np.convolve(signal, np.ones(window_size)/window_size, mode='same')    
        
        peaks, _ = scipy.signal.find_peaks(signal, height=0.005)  # Threshold height as needed

        heights= signal[peaks]
        peaks= peaks[ heights>threshold ]
        return peaks
    
    @staticmethod
    def linker( x, y ):
        """
            For a refrence sequence of points X, and a given set of points y, finds the likeliest linkage between the set of points.
            Then reporst numer of the missed points, and the list of displacements
        Parameters
        ----------
        x : Reference set of points
        y : given set of points

        Returns
        -------
            Missed points
            extra points
            average displacements

        """
        
        #link each poin in y to the neartest point in X
        closest_in_x=[]
        
        absent_in_y=np.ones(len(x))
        for i in range(len(y)):
            temp=np.zeros(len(x))
            for j in range(len(x)):
                temp[j] = np.abs( y[i]-x[j]  )
            # print("distances:", temp)
            k=np.argmin(temp)
            # print("closest to ",  y[i], "  is ",x[k] )
            closest_in_x.append(k)
            absent_in_y[k]=0
        # print(closest_in_x)
        absent_in_y=np.asarray(absent_in_y)
        missed_peaks= np.argwhere(absent_in_y==1)
        missed_peaks= missed_peaks[:,0] 
        extra_peaks_num=len(y)-len(x) +len(missed_peaks)

        return closest_in_x , missed_peaks,extra_peaks_num

