
from torchaudio.datasets import SPEECHCOMMANDS



import os, sys
sys.path.append(os.path.abspath(".."))

import matplotlib.pyplot as plt
import safari
import numpy as np
import tqdm
from tqdm import tqdm
import scipy.io as sio


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

train_set = SubsetSC("training")
test_set = SubsetSC("testing")




Walrus_params={'fname':'wavelet',
               'meas':'scaled',
               'wavelet_name':'db11',
               'L':2**19,
               'wavelet_scale_max':0,
               'wavelet_scale_min':-5,
               'wavelet_shift':0.01
               }
safari_walrus = safari.SSM(params=Walrus_params)

# build the frames for this experiment
hippo_legs = safari.SSM(params={'N':1995, 'fname':'legendre', 'meas':'scaled'})
safari_fous = safari.SSM(params={'N':1995, 'fname':'fourier', 'meas':'scaled'})






errs_wave=[]
errs_leg=[]
errs_fou=[]

Seq_len=50000


for x_ind in tqdm(range(800)):
    waveform, sample_rate, label, speaker_id, utterance_number = train_set[10000+x_ind]
    signal=waveform[0].numpy()    
    signal= safari.resample(signal,Seq_len)
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
    

    # plt.figure()
    # plt.plot(x           ,label='X')
    # plt.plot(xhat_waves  ,label='WaveS')
    # plt.plot(xhat_legs   ,label='LegS')
    # # plt.plot(xhat_fous[int(3*Seq_len/7):int(4*Seq_len/7)] ,label='FouS')
    # plt.legend()
    # plt.show()     
    
    errs_wave.append(np.linalg.norm(signal-xhat_waves))
    errs_leg.append(np.linalg.norm(signal-xhat_legs))
    errs_fou.append(np.linalg.norm(signal-xhat_fous))
    



errs_wave=np.asarray( errs_wave )
errs_leg=np.asarray( errs_leg )
errs_fou=np.asarray( errs_fou )




print("***************   Mean MSE comparison   ***************")
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





