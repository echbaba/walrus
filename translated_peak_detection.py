import os, sys
sys.path.append(os.path.abspath(".."))

import matplotlib.pyplot as plt
import safari
import numpy as np
import tqdm
from tqdm import tqdm
import scipy.io as sio

import warnings
# Ignore all warnings
warnings.filterwarnings('ignore')




Walrus_params={'fname':'wavelet',
               'meas':'translated',
               'wavelet_name':'db11',
               'L':2**16,
               'wavelet_scale_max':2,
               'wavelet_scale_min':0,
               'wavelet_shift':0.01
               }
safari_walrus = safari.SSM(params=Walrus_params)

# build the frames for this experiment
hippo_legt = safari.SSM(params={'N':65, 'fname':'legendre', 'meas':'translated'})
safari_fout = safari.SSM(params={'N':65, 'fname':'fourier', 'meas':'translated'})










    
Seq_len=10000
N_spikes=20
window_size=2000
evaluation_indices= np.arange( max( int(0.2*Seq_len) ,window_size) ,  Seq_len, window_size)
num_windows = len(evaluation_indices)





overall_missed_wave, overall_missed_leg, overall_missed_fou= [], [], []
overall_extra_wave, overall_extra_leg, overall_extra_fou= [], [], []
overall_disp_wave, overall_disp_leg, overall_disp_fou= [], [], []
overall_amp_wave, overall_amp_leg, overall_amp_fou=[], [], []


#Synthetic_modes=[ 'bumps', 'spikes']
Synthetic_mode='bumps'


for x_ind in tqdm(range(500)):
    #prepare the input signal
    heights=  np.random.normal(size=N_spikes) 
    widths=  0.005 + 0.005*np.random.uniform(size=N_spikes)
    
    if Synthetic_mode=='bumps':
        flag= True
        while( flag):
            flag=False
            locations=  np.sort( np.random.uniform( size=N_spikes) )
            for j in range(N_spikes-1):
                if(locations[j+1]-locations[j] < 0.75*(widths[j]+widths[j+1] ) ):
                    flag=True
        signal= safari.synthetic_signal.bumps(Seq_len, 2+np.abs(heights),locations, widths) 
    elif Synthetic_mode=='spikes':
        flag= True
        while( flag):
            flag=False
            locations=  np.sort( np.random.uniform( size=N_spikes) )
            for j in range(N_spikes-1):
                if(locations[j+1]-locations[j] < 0.005 ):
                    flag=True
        signal= safari.synthetic_signal.spikes(Seq_len,locations,2+ np.abs(heights), spike_width=0.002)
    signal=signal/ np.linalg.norm(signal)   
    signal_noiseless=1.0*signal
    noise=0.001*safari.resample( np.random.normal(size=2000)  , Seq_len)
    signal=signal+ noise
    
    collected_windows=np.asarray([])
    collected_xhat_wave=np.asarray([])
    collected_xhat_fou=np.asarray([])
    collected_xhat_leg=np.asarray([])
    
    #compute states
    c_legt= safari.compute_ssm_state( hippo_legt, signal ,W=window_size, eval_ind=evaluation_indices)
    c_fout = safari.compute_ssm_state( safari_fout, signal ,W=window_size, eval_ind=evaluation_indices)
    c_wavet = safari.compute_ssm_state( safari_walrus, signal ,W=window_size, eval_ind=evaluation_indices)

    running_MSE_wave, running_MSE_leg, running_MSE_fou= [], [], []
    for i in range(num_windows):
        current_window= signal_noiseless[  evaluation_indices[i]-window_size : evaluation_indices[i]   ]
        
        #find the recunstructions for the given computed state
        xhat_wavet = safari.reconstruct(c_wavet[:, i], window_size, safari_walrus.Fobj.D)
        xhat_legt = safari.reconstruct(c_legt[:, i], window_size, hippo_legt.Fobj.D)
        xhat_fout = safari.reconstruct(c_fout[:, i], window_size, safari_fout.Fobj.D)
        
        running_MSE_wave.append(    np.linalg.norm(current_window-xhat_wavet))
        running_MSE_leg.append(   np.linalg.norm(current_window-xhat_legt))
        running_MSE_fou.append(   np.linalg.norm(current_window-xhat_fout))
        
        collected_xhat_wave= np.concatenate(  (collected_xhat_wave, xhat_wavet ) )
        collected_xhat_leg = np.concatenate(  (collected_xhat_leg , xhat_legt ) )
        collected_xhat_fou = np.concatenate(  (collected_xhat_fou , xhat_fout)  )
        collected_windows  = np.concatenate(  (collected_windows  , current_window)  )
        
    #translate peaks location from [0,1] into indx in [0,Seq_len]
    x_peaks= ( Seq_len*locations).astype(int).squeeze()
    mask = (x_peaks < len(collected_windows))
    x_peaks=x_peaks[mask]

    collected_windows=collected_windows.squeeze()
    xhat_wave=collected_xhat_wave .squeeze()
    xhat_leg=collected_xhat_leg .squeeze()
    xhat_fou=collected_xhat_fou .squeeze()
    #now instead of having many windows, we have these.
    # collected_windows:concatenated x for all the processed windows 
    # xhat_wave: concatenated xhat_wave for all the processed windows
    # xhat_leg: concatenated xhat_leg for all the processed windows
    # xhat_fou: concatenated xhat_fou for all the processed windows
    
    
    
    #findings peaks in the reconstructed signals
    wave_peaks=  safari.peak_detection.peakfinder(xhat_wave, window_size=20, threshold=0.018)
    leg_peaks= safari.peak_detection.peakfinder(xhat_leg, window_size=20,  threshold=0.018)
    fou_peaks= safari.peak_detection.peakfinder(xhat_fou, window_size=20,  threshold=0.018)
    
    #count the missd and extra detected peaks
    linked_wave, missed_wave, extra_wave= safari.peak_detection.linker( x_peaks, wave_peaks )
    linked_leg, missed_leg, extra_leg= safari.peak_detection.linker( x_peaks, leg_peaks )
    linked_fou, missed_fou, extra_fou=  safari.peak_detection.linker( x_peaks, fou_peaks) 
    
    #measure the relative amplitude error for the detected peaks
    wave_amp_loss= np.abs(  xhat_wave[wave_peaks]-   signal_noiseless[ x_peaks[linked_wave ]  ]  ) / np.abs(   signal_noiseless[ x_peaks[linked_wave ]  ]  )
    leg_amp_loss= np.abs(  xhat_leg[leg_peaks]-   signal_noiseless[ x_peaks[linked_leg ]  ]  ) / np.abs(   signal_noiseless[ x_peaks[linked_leg ]  ]  )
    fou_amp_loss= np.abs(  xhat_fou[fou_peaks]-   signal_noiseless[ x_peaks[linked_fou ]  ]  ) / np.abs(   signal_noiseless[ x_peaks[linked_fou ]  ]  )
    
    #measure the displacement of detected peaks
    wave_displacement= np.abs( wave_peaks-x_peaks[linked_wave ] )
    leg_displacement= np.abs( leg_peaks-x_peaks[linked_leg ])
    fou_displacement= np.abs( fou_peaks-x_peaks[linked_fou ])
    
    N_peaks=len(x_peaks)
    overall_missed_wave.append(  len(missed_wave)/N_peaks)
    overall_missed_fou.append( len(missed_fou)/N_peaks)
    overall_missed_leg.append( len(missed_leg)/N_peaks)
    
    overall_extra_wave.append(  extra_wave/N_peaks)
    overall_extra_fou.append( extra_fou/N_peaks)
    overall_extra_leg.append( extra_leg/N_peaks)
    
    
    overall_disp_wave.extend( wave_displacement )
    overall_disp_fou.extend( fou_displacement )
    overall_disp_leg.extend( leg_displacement )
    
    overall_amp_wave.extend( wave_amp_loss )
    overall_amp_fou.extend(  fou_amp_loss )
    overall_amp_leg.extend(  leg_amp_loss )
    
    # #Uncomment this  to have every instance where walrus is outperformed plotted.
    # if( len(missed_wave) > len(missed_fou) or len(missed_wave) > len(missed_leg) ):
    #     plt.figure()
    #     plt.plot(signal_noiseless,label='XGT')
    #     plt.scatter(x_peaks,  signal_noiseless[x_peaks], color='blue',  zorder=3)
    #     plt.plot(xhat_wave,label='db')
    #     plt.plot(xhat_leg,label='leg')
    #     plt.plot(xhat_fou,label='fou')
    #     plt.scatter(wave_peaks,  xhat_wave[wave_peaks], color='orange',  zorder=3)
    #     plt.legend()
    #     plt.show()







overall_missed_wave= np.asarray(overall_missed_wave)
overall_missed_leg= np.asarray(overall_missed_leg)
overall_missed_fou= np.asarray(overall_missed_fou)





print("wave miss= ", np.mean(overall_missed_wave ))
print("leg miss=", np.mean(overall_missed_leg ))
print("fou miss= ",  np.mean(overall_missed_fou ))


print("wave extra= ", np.mean(overall_extra_wave ))
print("leg extra=", np.mean(overall_extra_leg ))
print("fou extra= ",  np.mean(overall_extra_fou ))

data_amp = [
        np.log(overall_amp_leg).tolist(),
        np.log(overall_amp_fou).tolist(),
        np.log(overall_amp_wave).tolist()
    ]

data_disp= [
        np.log(overall_disp_leg).tolist(),
        np.log(overall_disp_fou).tolist(),
        np.log(overall_disp_wave).tolist()
    ]

plt.boxplot(data_amp)
plt.xticks([1, 2, 3], ['leg', 'fou', 'wave'])
plt.xlabel("Columns")
plt.ylabel("log(Values)")
plt.title("Amplitude of the captured peaks")
plt.show()
    
plt.boxplot(data_disp)
plt.xticks([1, 2, 3], ['leg', 'fou', 'wave'])
plt.xlabel("Columns")
plt.ylabel("log(Values)")
plt.title("Disp of the captured peaks")
plt.show()
    



wave_win= np.ones(len(overall_missed_wave))
wave_win[  np.argwhere( overall_missed_wave>overall_missed_leg) ]=0
wave_win[  np.argwhere( overall_missed_wave>overall_missed_fou) ]=0
print(" db win percentage:",  np.mean(wave_win))

fou_win= np.ones(len(overall_missed_fou))
fou_win[  np.argwhere( overall_missed_fou>overall_missed_wave) ]=0
fou_win[  np.argwhere( overall_missed_fou>overall_missed_leg) ]=0
print(" fou win percentage:",  np.mean(fou_win))


leg_win= np.ones(len(overall_missed_leg))
leg_win[ np.argwhere( overall_missed_leg>overall_missed_fou) ]=0
leg_win[ np.argwhere( overall_missed_leg>overall_missed_wave) ]=0
print(" leg win percentage:",  np.mean(leg_win))





mdic={ 'wave_missed':overall_missed_wave, 'wave_disp': overall_disp_wave,'wave_amp': overall_amp_wave 
      ,'leg_missed':overall_missed_leg, 'leg_disp': overall_disp_leg,'leg_amp': overall_amp_leg 
      ,'fou_missed':overall_missed_fou, 'fou_disp': overall_disp_fou,'fou_amp': overall_amp_fou
      ,'wave_extra':overall_extra_wave , 'leg_extra':overall_extra_leg, 'fou_extra':overall_extra_fou
      }


print("Rel amp error:")
print("Leg:", 100*np.mean(overall_amp_leg), "%")
print("Fou:", 100*np.mean(overall_amp_fou), "%")
print("Wave:", 100*np.mean(overall_amp_wave), "%")


print("Displacement:")
print("Leg:", np.mean(overall_disp_leg))
print("Fou:", np.mean(overall_disp_fou))
print("Wave:", np.mean(overall_disp_wave))


# sio.savemat('ExpResults/Exp_NLA_translated_Spikes.mat', mdic)


