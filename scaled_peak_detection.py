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
               'wavelet_scale_max':3,
               'wavelet_scale_min':0,
               'wavelet_shift':0.01
               }
safari_walrus = safari.SSM(params=Walrus_params)

# build the frames for this experiment
hippo_legs = safari.SSM(params={'N':65, 'fname':'legendre', 'meas':'scaled'})
safari_fous = safari.SSM(params={'N':65, 'fname':'fourier', 'meas':'scaled'})




#creating the lists to collect hte relevant statistics during the experiments
overall_missed_wave, overall_missed_leg, overall_missed_fou = [], [], []
overall_extra_wave, overall_extra_leg, overall_extra_fou= [], [], []
overall_disp_wave, overall_disp_leg, overall_disp_fou=[], [], []
overall_amp_wave, overall_amp_leg, overall_amp_fou =[], [], []

#Synthetic_modes=[ 'bumps', 'spikes']
Synthetic_mode='spikes'
Seq_len=10000
N_spikes=10

for x_ind in tqdm(range(100)):
    
    #Creat random instances of puleses or spikes
    heights=  np.random.normal(size=N_spikes) 
    widths=  0.02 + 0.015*np.random.uniform(size=N_spikes)
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
                if(locations[j+1]-locations[j] < 0.03):
                    flag=True
        signal= safari.synthetic_signal.spikes(Seq_len,locations,2+ np.abs(heights), spike_width=0.01)
        
    signal=signal/ np.linalg.norm(signal)
    signal_noiseless=1.0*signal
    noise=0.001*safari.resample( np.random.normal(size=int(0.2*Seq_len))  , Seq_len )  #Creating noise
    signal=signal+ noise


    #compute states
    c_legs= safari.compute_ssm_state( hippo_legs, signal )
    c_fous = safari.compute_ssm_state( safari_fous, signal )
    c_waves = safari.compute_ssm_state( safari_walrus, signal )
    
    #find the recunstructions for the given computed state
    xhat_legs = safari.reconstruct(c_legs[:,-1], Seq_len, hippo_legs.Fobj.D)
    xhat_fous = safari.reconstruct(c_fous[:,-1], Seq_len, safari_fous.Fobj.D)
    xhat_waves = safari.reconstruct(c_waves[:,-1], Seq_len, safari_walrus.Fobj.D)

    
    signal_noiseless=signal_noiseless.squeeze()
    signal_noisy=signal.squeeze()
    xhat_waves=xhat_waves .squeeze()
    xhat_legs=xhat_legs .squeeze()
    xhat_fous=xhat_fous .squeeze()
    
    #translate peaks location from [0,1] into indx in [0,Seq_len]
    x_peaks=(Seq_len* locations).astype(int).squeeze()
    #now instead of having many windows, we have these.
    # collected_windows:concatenated x for all the processed windows 
    # xhat_wave: concatenated xhat_wave for all the processed windows
    # xhat_leg: concatenated xhat_leg for all the processed windows
    # xhat_fou: concatenated xhat_fou for all the processed windows
    
    
    #findings peaks in the reconstructed signals
    wave_peaks=safari.peak_detection.peakfinder(xhat_waves,  window_size=100, threshold=0.007)
    leg_peaks=safari.peak_detection.peakfinder(xhat_legs,  window_size=100, threshold=0.007)
    fou_peaks=safari.peak_detection.peakfinder(xhat_fous,  window_size=100, threshold=0.007)
    
    #count the missd and extra detected peaks
    linked_wave, missed_wave, extra_wave = safari.peak_detection.linker( x_peaks, wave_peaks )
    linked_leg, missed_leg, extra_leg= safari.peak_detection.linker( x_peaks, leg_peaks )
    linked_fou, missed_fou, extra_fou= safari.peak_detection.linker( x_peaks, fou_peaks ) 
    
    #measure the relative amplitude error for the detected peaks
    wave_amp_loss= np.abs(  xhat_waves[wave_peaks]-   signal_noiseless[ x_peaks[linked_wave ]  ]  ) / np.abs(   signal_noiseless[ x_peaks[linked_wave ]  ]  )
    leg_amp_loss= np.abs(  xhat_legs[leg_peaks]-   signal_noiseless[ x_peaks[linked_leg ]  ]  ) / np.abs(   signal_noiseless[ x_peaks[linked_leg ]  ]  )
    fou_amp_loss= np.abs(  xhat_fous[fou_peaks]-   signal_noiseless[ x_peaks[linked_fou ]  ]  ) / np.abs(   signal_noiseless[ x_peaks[linked_fou ]  ]  )
    
    #measure the displacement of detected peaks
    wave_displacement= np.abs( wave_peaks-x_peaks[linked_wave ] )
    leg_displacement= np.abs( leg_peaks-x_peaks[linked_leg ])
    fou_displacement= np.abs( fou_peaks-x_peaks[linked_fou ])
    
    overall_missed_wave.append(  len(missed_wave)/N_spikes)
    overall_missed_fou.append( len(missed_fou)/N_spikes)
    overall_missed_leg.append( len(missed_leg)/N_spikes)
    
    overall_extra_wave.append(  extra_wave/N_spikes)
    overall_extra_fou.append( extra_fou/N_spikes)
    overall_extra_leg.append( extra_leg/N_spikes)
    
    
    overall_disp_wave.extend( wave_displacement )
    overall_disp_fou.extend( fou_displacement )
    overall_disp_leg.extend( leg_displacement )
    overall_amp_wave.extend(  wave_amp_loss )
    overall_amp_fou.extend(  fou_amp_loss )
    overall_amp_leg.extend(  leg_amp_loss )
    
    
    #uncomment this to visualize instances where another SSM has less missed peaks than SaFARi-WaveS
    # if( len(missed_wave) > len(missed_fou) or len(missed_wave) > len(missed_leg) ):
    #     plt.figure()
    #     plt.plot(x_noiseless,label='XGT')
    #     plt.plot(xhat_waves,label='db')
    #     plt.plot(xhat_legs,label='leg')
    #     plt.plot(xhat_fous,label='fou')
    #     plt.legend()
    #     plt.show()
        



overall_missed_wave= np.asarray(overall_missed_wave)
overall_missed_leg= np.asarray(overall_missed_leg)
overall_missed_fou= np.asarray(overall_missed_fou)


print("WaveS miss= ", np.mean(overall_missed_wave ))
print("LegS miss=", np.mean(overall_missed_leg ))
print("FouS miss= ",  np.mean(overall_missed_fou ))

print("WaveS extra= ", np.mean(overall_extra_wave ))
print("LegS extra=", np.mean(overall_extra_leg ))
print("FouS extra= ",  np.mean(overall_extra_fou ))


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
plt.xticks([1, 2, 3], ['LegS', 'FouS', 'WaveS'])
plt.xlabel("Columns")
plt.ylabel("log(relative_amp)")
plt.title("Relative amplitude error of the captured peaks")
plt.show()
    
plt.boxplot(data_disp)
plt.xticks([1, 2, 3], ['LegS', 'FouS', 'Waves'])
plt.xlabel("Columns")
plt.ylabel("log(dislacement)")
plt.title("Displacement of the captured peaks")
plt.show()
    


print(" ######### Number of instances where each ssm has lowest missed peaks #############")
wave_win= np.ones(len(overall_missed_wave))
wave_win[  np.argwhere( overall_missed_wave>overall_missed_leg) ]=0
wave_win[  np.argwhere( overall_missed_wave>overall_missed_fou) ]=0
print(" WaveS win percentage:",  np.mean(wave_win))

fou_win= np.ones(len(overall_missed_fou))
fou_win[  np.argwhere( overall_missed_fou>overall_missed_wave) ]=0
fou_win[  np.argwhere( overall_missed_fou>overall_missed_leg) ]=0
print(" fouS win percentage:",  np.mean(fou_win))


leg_win= np.ones(len(overall_missed_leg))
leg_win[ np.argwhere( overall_missed_leg>overall_missed_fou) ]=0
leg_win[ np.argwhere( overall_missed_leg>overall_missed_wave) ]=0
print(" legS win percentage:",  np.mean(leg_win))





mdic={ 'wave_missed':overall_missed_wave, 'wave_disp': overall_disp_wave,'wave_amp': overall_amp_wave 
      ,'leg_missed':overall_missed_leg, 'leg_disp': overall_disp_leg,'leg_amp': overall_amp_leg 
      ,'fou_missed':overall_missed_fou, 'fou_disp': overall_disp_fou,'fou_amp': overall_amp_fou
      ,'wave_extra':overall_extra_wave , 'leg_extra':overall_extra_leg, 'fou_extra':overall_extra_fou
      }

# # sio.savemat('ExpResults/Exp_NLA_spikes.mat', mdic)

print("\n######### relative amplitude error #############")
print("LegS:", 100*np.mean(overall_amp_leg), "%")
print("FouS:", 100*np.mean(overall_amp_fou), "%")
print("WaveS:", 100*np.mean(overall_amp_wave), "%")



print("\n######### relative displacement error #############")
print("LegS:", np.mean(overall_disp_leg))
print("FouS:", np.mean(overall_disp_fou))
print("Waves:", np.mean(overall_disp_wave))





