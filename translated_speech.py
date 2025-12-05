import os, sys
sys.path.append(os.path.abspath(".."))

import matplotlib.pyplot as plt
import safari
import numpy as np
import tqdm
from tqdm import tqdm
import scipy.io as sio
import pandas as pd
from torchaudio.datasets import SPEECHCOMMANDS



#Define a subset loader for the SpeechCommands dataset
class SubsetSpeechCommands(SPEECHCOMMANDS):
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


# Load training and testing datasets
train_dataset = SubsetSpeechCommands(subset="training")
test_dataset = SubsetSpeechCommands(subset="testing")



Walrus_params={'fname':'wavelet',
               'meas':'translated',
               'wavelet_name':'db11',
               'L':2**18,
               'wavelet_scale_max':1,
               'wavelet_scale_min':-3,
               'wavelet_shift':0.0025
               }
safari_walrus = safari.SSM(params=Walrus_params)

# build the frames for this experiment
hippo_legt = safari.SSM(params={'N':500, 'fname':'legendre', 'meas':'translated'})
safari_fout = safari.SSM(params={'N':500, 'fname':'fourier', 'meas':'translated'})






# Experiment parameters
window_size = 30000           # Size of the translated window (T0)
sequence_length = 64000       # Resampled signal length
step_size = window_size // 20
evaluation_indices = np.arange(
    max(int(0.2 * sequence_length), window_size),
    sequence_length,
    step_size
)

# Initialize error storage arrays
num_trials = 1000
num_windows = len(evaluation_indices)

errors_wavelet = np.zeros((num_trials, num_windows))
errors_legendre = np.zeros((num_trials, num_windows))
errors_fourier = np.zeros((num_trials, num_windows))


# Main evaluation loop
for trial_idx in tqdm(range(num_trials)):
    # Load and preprocess a waveform from the dataset
    waveform, sample_rate, label, speaker_id, utterance_number = train_dataset[10000 + trial_idx]
    signal = waveform[0].numpy()
    signal = safari.resample(signal, sequence_length)
    signal /= np.linalg.norm(signal)

    # Compute SSM representations for each frame type
    coeffs_wavelet = safari.compute_ssm_state(safari_walrus, signal, W=window_size, eval_ind=evaluation_indices)
    coeffs_legendre = safari.compute_ssm_state(hippo_legt, signal, W=window_size, eval_ind=evaluation_indices)
    coeffs_fourier = safari.compute_ssm_state(safari_fout, signal, W=window_size, eval_ind=evaluation_indices)

    # Store running MSE for each window
    mse_wavelet, mse_legendre, mse_fourier = [], [], []

    for window_idx in range(num_windows):
        # Extract current window from the signal
        current_window = signal[evaluation_indices[window_idx] - window_size : evaluation_indices[window_idx]]

        # Reconstruct signals from SSM coefficients
        recon_wavelet = safari.reconstruct(coeffs_wavelet[:, window_idx], window_size, safari_walrus.Fobj.D)
        recon_legendre = safari.reconstruct(coeffs_legendre[:, window_idx], window_size, hippo_legt.Fobj.D)
        recon_fourier = safari.reconstruct(coeffs_fourier[:, window_idx], window_size, safari_fout.Fobj.D)

        # Compute normalized reconstruction errors
        mse_wavelet.append(np.linalg.norm(recon_wavelet - current_window) / np.linalg.norm(current_window))
        mse_legendre.append(np.linalg.norm(recon_legendre - current_window) / np.linalg.norm(current_window))
        mse_fourier.append(np.linalg.norm(recon_fourier - current_window) / np.linalg.norm(current_window))

        # Uncomment to visualize reconstructed signals
        # plt.figure()
        # plt.plot(current_window[2000:8000])
        # plt.plot(recon_legendre[2000:8000], label='Legendre')
        # plt.plot(recon_wavelet[2000:8000], label='Wavelet')
        # plt.legend()
        # plt.show()

    # Store errors for this trial
    errors_wavelet[trial_idx, :] = np.array(mse_wavelet)
    errors_legendre[trial_idx, :] = np.array(mse_legendre)
    errors_fourier[trial_idx, :] = np.array(mse_fourier)

    # # Uncomment to plot running log MSE across windows
    # plt.figure()
    # plt.plot(np.log(mse_legendre), label='Legendre')
    # plt.plot(np.log(mse_fourier), label='Fourier')
    # plt.plot(np.log(mse_wavelet), label='Wavelet')
    # plt.legend()
    # plt.show()

t= np.arange(num_windows)

plt.figure()
plt.fill_between(t,   np.log(np.quantile(errors_wavelet, 0.4 , axis=0 )), np.log(np.quantile(errors_wavelet, 0.6 , axis=0 ))  , alpha=0.3 , label='wave', color='green')
plt.fill_between(t,  np.log( np.quantile(errors_legendre, 0.4 , axis=0 )), np.log( np.quantile(errors_legendre, 0.6 , axis=0 ))  , alpha=0.3 , label='leg', color='blue')
plt.fill_between(t,   np.log(np.quantile(errors_fourier, 0.4 , axis=0 )), np.log(np.quantile(errors_fourier, 0.6 , axis=0 ))  , alpha=0.3, label='fou', color='orange')
plt.plot( np.log( np.median(errors_wavelet, 0)   ), label='wave', color='green' )
plt.plot( np.log( np.median(errors_legendre , 0)    ), label='leg', color='blue'  )
plt.plot( np.log( np.median(errors_fourier , 0)    ), label='fou', color='orange'  )

plt.legend()
plt.show()





# #save as csv file
# df = pd.DataFrame({
#     'ind': np.arange(num_windows),
#     'db_med':  np.median(errors_wavelet, axis=0 ) ,
#     'db_low':  np.quantile(errors_wavelet, 0.4 , axis=0 ) ,
#     'db_high': np.quantile(errors_wavelet, 0.6 , axis=0 ),
    
#     'leg_med':  np.median(errors_legendre, axis=0 ) ,
#     'leg_low':  np.quantile(errors_legendre, 0.4 , axis=0 ) ,
#     'leg_high': np.quantile(errors_legendre, 0.6 , axis=0 ),
    
#     'fou_med':  np.median(errors_fourier, axis=0 ) ,
#     'fou_low':  np.quantile(errors_fourier, 0.4 , axis=0 ) ,
#     'fou_high': np.quantile(errors_fourier, 0.6 , axis=0 ),
# })
# # Save to CSV
# df.to_csv('Frames/translated_speech.csv', index=False)




# Uncomment to save the experiment results as .mat file
# results = {
#     'errs_wavelet': errors_wavelet,
#     'errs_legendre': errors_legendre,
#     'errs_fourier': errors_fourier
# }
# sio.savemat('ExpResults/Exp_translated_speech.mat', results)