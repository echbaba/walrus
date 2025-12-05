

import os, sys
sys.path.append(os.path.abspath(".."))

import matplotlib.pyplot as plt
import safari
import numpy as np
import tqdm
from tqdm import tqdm
import scipy.io as sio



Walrus_params={'fname':'wavelet',
               'meas':'scaled',
               'wavelet_name':'db11',
               'L':2**15,
               'wavelet_scale_max':2,
               'wavelet_scale_min':-3,
               'wavelet_shift':0.01
               }
safari_walrus = safari.SSM(params=Walrus_params)

# build the frames for this experiment
hippo_legs = safari.SSM(params={'N':500, 'fname':'legendre', 'meas':'scaled'})
safari_fous = safari.SSM(params={'N':500, 'fname':'fourier', 'meas':'scaled'})








    
Seq_len=10000
N_spikes=1000


errs_wave=[]
errs_leg=[]
errs_fou=[]

#Synthetic_modes=['blocks', 'bumps', 'piecepoly', 'spikes']
Synthetic_mode='spikes'

for x_ind in tqdm(range(1000)):
    #prepare the inout signal
    heights=  np.random.normal(size=N_spikes) 
    widths=  0.001 + 0.0005*np.random.uniform(size=N_spikes)
    locations=  np.sort( np.random.uniform( size=N_spikes) )
    if Synthetic_mode== 'blocks':
        signal=safari.synthetic_signal.blocks(Seq_len,heights,locations) 
    elif Synthetic_mode=='bumps':
        signal=safari.synthetic_signal.bumps(Seq_len,np.abs(heights),locations, widths) 
    elif Synthetic_mode=='piecepoly':
        signal= safari.synthetic_signal.piecewise_poly(Seq_len)
    elif Synthetic_mode=='spikes':
        signal=safari.synthetic_signal.spikes(Seq_len,locations, np.abs(heights), spike_width=0.001)
        
    signal=signal/ np.linalg.norm(signal)
     
    #Compute the states 
    c_legs= safari.compute_ssm_state( hippo_legs, signal )
    c_fous = safari.compute_ssm_state( safari_fous, signal )
    c_waves = safari.compute_ssm_state( safari_walrus, signal )
    
    #find the recunstructions for the given computed state
    xhat_legs = safari.reconstruct(c_legs[:,-1], Seq_len, hippo_legs.Fobj.D)
    xhat_fous = safari.reconstruct(c_fous[:,-1], Seq_len, safari_fous.Fobj.D)
    xhat_waves = safari.reconstruct(c_waves[:,-1], Seq_len, safari_walrus.Fobj.D)
    
    #collect the reconstruction errors
    errs_wave.append(np.linalg.norm(signal-xhat_waves))
    errs_leg.append(np.linalg.norm(signal-xhat_legs))
    errs_fou.append(np.linalg.norm(signal-xhat_fous))
    


    

errs_wave=np.asarray( errs_wave )
errs_leg=np.asarray( errs_leg )
errs_fou=np.asarray( errs_fou )



print("***************   average MSE comparison   ***************")
print( "WaveS: ", np.mean(errs_wave), "          std:", np.std(errs_wave) )
print( "LegS: ", np.mean(errs_leg), "          std:", np.std(errs_leg)  )
print( "FouS: ", np.mean(errs_fou), "          std:", np.std(errs_fou)  )





wave_wins= 1.0*( errs_wave- errs_leg <=0 )*(errs_wave- errs_fou<=0)
leg_wins= 1.0*( errs_leg- errs_wave <=0 )*(errs_leg- errs_fou<=0)
fou_wins= 1.0*( errs_fou- errs_wave <=0 )*(errs_fou- errs_leg<=0)
  

print("***************   Percentage of instances where an ssm has lowest MSE   ***************")
print( "WaveS: ", np.mean(wave_wins) )
print( "LegS: ", np.mean(leg_wins) )
print( "FouS: ", np.mean(fou_wins) )


