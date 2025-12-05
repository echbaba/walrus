import os, sys
sys.path.append(os.path.abspath(".."))

import matplotlib.pyplot as plt
import safari
import numpy as np
import tqdm
from tqdm import tqdm
import scipy.io as sio

#import data
M4File=sio.loadmat('../Datasets/M4/Monthly-train_0.mat')
inputs= M4File['data']

Walrus_params={'fname':'wavelet',
               'meas':'scaled',
               'wavelet_name':'db11',
               'L':2**18,
               'wavelet_scale_max':2,
               'wavelet_scale_min':-3,
               'wavelet_shift':0.01
               }
safari_walrus = safari.SSM(params=Walrus_params)

# build the frames for this experiment
hippo_legs = safari.SSM(params={'N':500, 'fname':'legendre', 'meas':'scaled'})
safari_fous = safari.SSM(params={'N':500, 'fname':'fourier', 'meas':'scaled'})



err_leg, err_fou, err_wave=[], [], []
for x_ind in range(300):

    signal=inputs[:,x_ind] 
    Seq_len=5000
    signal= safari.resample(signal,Seq_len)
    signal=signal/ np.linalg.norm(signal)


    #Compute the states 
    c_legs= safari.compute_ssm_state( hippo_legs, signal )
    c_fous = safari.compute_ssm_state( safari_fous, signal )
    c_walrus = safari.compute_ssm_state( safari_walrus, signal )

    #find the recunstructions for the given computed state
    xhat_legs = safari.reconstruct(c_legs[:,-1], Seq_len, hippo_legs.Fobj.D)
    xhat_fous = safari.reconstruct(c_fous[:,-1], Seq_len, safari_fous.Fobj.D)
    xhat_walrus = safari.reconstruct(c_walrus[:,-1], Seq_len, safari_walrus.Fobj.D)

    err_wave.append( np.linalg.norm(signal-xhat_walrus) )
    err_fou.append( np.linalg.norm(signal-xhat_fous) )
    err_leg.append( np.linalg.norm(signal-xhat_legs) )
    

err_wave=np.asarray( err_wave )
err_leg=np.asarray( err_leg )
err_fou=np.asarray( err_fou )

print("***************   Mean MSE comparison   ***************")
print( "WaveS: ", np.mean(err_wave), "          std:", np.std(err_wave) )
print( "LegS: ", np.mean(err_leg), "          std:", np.std(err_leg)  )
print( "FouS: ", np.mean(err_fou), "          std:", np.std(err_fou)  )



wave_wins= 1.0*( err_wave- err_leg <=0 )*(err_wave- err_fou<=0)
leg_wins= 1.0*( err_leg- err_wave <=0 )*(err_leg- err_fou<=0)
fou_wins= 1.0*( err_fou- err_wave <=0 )*(err_fou- err_leg<=0)
  

print("***************   Percentage of instances where an ssm has lowest MSE   ***************")
print( "WaveS: ", np.mean(wave_wins) )
print( "LegS: ", np.mean(leg_wins) )
print( "FouS: ", np.mean(fou_wins) )


