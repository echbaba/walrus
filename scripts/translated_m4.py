import os, sys
sys.path.append(os.path.abspath(".."))

import matplotlib.pyplot as plt
import safari
import numpy as np
import tqdm
from tqdm import tqdm
import scipy.io as sio
import pandas as pd

Walrus_params={'fname':'wavelet',
               'meas':'translated',
               'wavelet_name':'db11',
               'L':2**18,
               'wavelet_scale_max':2,
               'wavelet_scale_min':-1,
               'wavelet_shift':0.01
               }
safari_walrus = safari.SSM(params=Walrus_params)

# build the frames for this experiment
hippo_legt = safari.SSM(params={'N':128, 'fname':'legendre', 'meas':'translated'})
safari_fout = safari.SSM(params={'N':128, 'fname':'fourier', 'meas':'translated'})




M4File=sio.loadmat('../Datasets/M4/Monthly-train_0.mat')
inputs= M4File['data']
window_size=1000
Seq_len=8000
num_trials= 4000

evaluation_indices= np.arange( max( int(0.2*Seq_len) ,window_size) ,  Seq_len, 200)
num_windows = len(evaluation_indices)

errs_leg= np.zeros((num_trials,num_windows))
errs_fou= np.zeros((num_trials,num_windows))
errs_wave= np.zeros((num_trials,num_windows))


for x_ind in tqdm(range(num_trials)):
    
    #preparing the input
    signal=inputs[:,x_ind]
    signal= safari.resample(signal,Seq_len)
    signal=signal/ np.linalg.norm(signal)
    

    mse_wavelet, mse_legendre, mse_fourier = [], [], []
    
    collected_windows=np.asarray([])
    collected_recon_wave=np.asarray([])
    collected_recon_fou=np.asarray([])
    collected_recon_leg=np.asarray([])

    # Compute SSM representations for each frame type
    coeffs_wavelet = safari.compute_ssm_state(safari_walrus, signal, W=window_size, eval_ind=evaluation_indices)
    coeffs_legendre = safari.compute_ssm_state(hippo_legt, signal, W=window_size, eval_ind=evaluation_indices)
    coeffs_fourier = safari.compute_ssm_state(safari_fout, signal, W=window_size, eval_ind=evaluation_indices)
    
    
    base=np.linalg.norm(signal  )* Seq_len / window_size 
    for window_idx in range(num_windows):
        # Extract current window from the signal
        current_window = signal[evaluation_indices[window_idx] - window_size : evaluation_indices[window_idx]]

        # Reconstruct signals from SSM coefficients
        recon_wavelet = safari.reconstruct(coeffs_wavelet[:, window_idx], window_size, safari_walrus.Fobj.D)
        recon_legendre = safari.reconstruct(coeffs_legendre[:, window_idx], window_size, hippo_legt.Fobj.D)
        recon_fourier = safari.reconstruct(coeffs_fourier[:, window_idx], window_size, safari_fout.Fobj.D)

        # Compute normalized reconstruction errors
        mse_wavelet.append(np.linalg.norm(recon_wavelet - current_window))
        mse_legendre.append( np.linalg.norm(recon_legendre - current_window)  )
        mse_fourier.append(np.linalg.norm(recon_fourier - current_window) )
        

        collected_recon_wave= np.concatenate(  (collected_recon_wave, recon_wavelet ) )
        collected_recon_leg = np.concatenate(  (collected_recon_leg , recon_legendre ) )
        collected_recon_fou = np.concatenate(  (collected_recon_fou , recon_fourier)  )
        collected_windows  = np.concatenate(  (collected_windows  , current_window)  )
        
    
    errs_wave[x_ind,:]= np.asarray(mse_wavelet)
    errs_leg[x_ind,:]= np.asarray(mse_legendre)
    errs_fou[x_ind,:]= np.asarray(mse_fourier)
    
    
    
    # uncomment this if you want to plot concatenation of all reconstructed window appended
    # plt.figure()
    # plt.plot(collected_windows, label='signal')
    # plt.plot(collected_recon_wave, label='wave')
    # plt.plot(collected_recon_leg, label='leg')
    # # plt.plot(collected_recon_fou, label='fou')
    # plt.legend()
    # plt.show()
    
    # plt.figure()
    # plt.plot(mse_wavelet, label='wave')
    # plt.plot(mse_legendre, label='leg')
    # plt.legend()
    # plt.show()
    

    
t= np.arange(num_windows)
plt.figure()
plt.fill_between(t,   np.log(np.quantile(errs_wave, 0.4 , axis=0 )), np.log(np.quantile(errs_wave, 0.6 , axis=0 ))  , alpha=0.25 , label='wave', color='green')
plt.fill_between(t,  np.log( np.quantile(errs_leg, 0.4 , axis=0 )), np.log( np.quantile(errs_leg, 0.6 , axis=0 ))  , alpha=0.25 , label='leg', color='blue')
plt.fill_between(t,   np.log(np.quantile(errs_fou, 0.4 , axis=0 )), np.log(np.quantile(errs_fou, 0.6 , axis=0 ))  , alpha=0.5, label='fou', color='red')
plt.plot(np.log( np.median(errs_wave, 0)   ), label='wave', color='green')
plt.plot(np.log( np.median(errs_leg , 0)    ), label='leg', color='blue')
plt.plot(np.log( np.median(errs_fou , 0)    ), label='leg', color='red')
plt.legend()
plt.show()



# #save the data for the plot
# df = pd.DataFrame({
#     'ind': np.arange(num_windows),
#     'db_med':  np.median(errs_wave, axis=0 ) ,
#     'db_low':  np.quantile(errs_wave, 0.4 , axis=0 ) ,
#     'db_high': np.quantile(errs_wave, 0.6 , axis=0 ),
    
#     'leg_med':  np.median(errs_leg, axis=0 ) ,
#     'leg_low':  np.quantile(errs_leg, 0.4 , axis=0 ) ,
#     'leg_high': np.quantile(errs_leg, 0.6 , axis=0 ),
    
#     'fou_med':  np.median(errs_fou, axis=0 ) ,
#     'fou_low':  np.quantile(errs_fou, 0.4 , axis=0 ) ,
#     'fou_high': np.quantile(errs_fou, 0.6 , axis=0 ),
# })

# # Save to CSV
# df.to_csv('Frames/translated_m4.csv', index=False)
