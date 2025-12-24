import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
import os
from scipy.integrate import trapezoid

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq=.5*fs
    low=lowcut/nyq 
    high=highcut/nyq
    b, a=butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def main(eeg_file="my_eeg.csv"):

    if eeg_file and os.path.exists(eeg_file):
        eeg_signal = np.loadtxt(eeg_file)
        fs = 250  # adjust if your file has a different sampling rate
        t = np.arange(0, len(eeg_signal)/fs, 1/fs)
    else:
        # simulated EEG
        fs = 250
        t = np.arange(0, 10, 1/fs)
        delta = np.sin(2 * np.pi * 2 * t)
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t)
        noise = 0.3 * np.random.randn(len(t))
        eeg_signal = delta + alpha + noise
        
    filtered=bandpass_filter(eeg_signal, 1, 40, fs)
        
    plt.figure()
    plt.plot(t, filtered)
    plt.title("Filtered EEG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig("filtered_signal.png")
    plt.show()
        
    freqs, psd = welch(filtered, fs)
    
    plt.figure()
    plt.semilogy(freqs, psd)
    plt.title("EEG Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.savefig("psd_plot.png")
    plt.show()
    
    bands = {   
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30)  
    }
    
    band_powers = {}
    
    for band, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_powers[band] = trapezoid(psd[idx], freqs[idx])
        
    print("Band Powers:")
    for band, power in band_powers.items():
        print(f"{band}: {power:.3f}")
    
if __name__=="__main__":
    filename = input("Enter EEG CSV filename: ")
    main(filename)