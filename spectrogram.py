import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ddd_vector_length(data_mtx):
    return np.sqrt(np.square(data_mtx[:, 0]) + np.square(data_mtx[:, 1]) + np.square(data_mtx[:, 2]))


# in_path = input('Adatok elérési útja (mappa): ') # két file helye
# original = in_path + '/' + input('eredeti faljnev (+.csv): ')


file = "A://Machine Learning//project1//vector_lenght_test//rot.csv"
data = pd.read_csv(file, header= None)  # elforgatott fájl beolvasása
acceleration = data.as_matrix(columns=[1, 2, 3])

acc_vect_length = ddd_vector_length(acceleration)


figure, axarr = plt.subplots(2, sharex=True)
# axarr[0].plot(acceleration[:, 0])
axarr[0].plot(acc_vect_length)

NFFT = 50 # the length of the windowing segments
Fs = 1  # the sampling frequency
noverlap = 46


# axarr[1].specgram(acceleration[:, 0], cmap='plasma', NFFT=NFFT, Fs=Fs, noverlap=noverlap)
axarr[1].specgram(acc_vect_length, cmap='plasma', NFFT=NFFT, Fs=Fs, noverlap=noverlap)


figure.show()
plt.pause(900000)


input('Press key to end...')

