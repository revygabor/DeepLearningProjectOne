import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rotated_1 = "data_dani.csv"
data_1 = pd.read_csv(rotated_1, header= None).as_matrix(columns=[0, 1, 2])

rotated_2 = "data_gabor.csv"
data_2 = pd.read_csv(rotated_2, header= None).as_matrix(columns=[0, 1, 2])

figure, axarr = plt.subplots(2, sharex=True)

for i in range(200):#data_1.shape[0]-50):
    axarr[0].plot(np.fft.rfftfreq(100, 1/50), np.abs(np.fft.rfft(data_1[i:i+101, 0])), 'r', alpha=0.2)
# axarr[0].plot(np.fft.rfftfreq(50, 1/50), np.abs(np.fft.rfft(data_1[527:527+51, 0])))
# axarr[0].plot(np.fft.rfftfreq(50, 1/50), np.abs(np.fft.rfft(data_1[927:927+51, 0])))

for i in range(200):#data_2.shape[0]-50):
    axarr[1].plot(np.fft.rfftfreq(100, 1/50), np.abs(np.fft.rfft(data_2[i:i+101, 0])), 'r', alpha=0.2)
# axarr[1].plot(np.fft.rfftfreq(50, 1/50), np.abs(np.fft.rfft(data_2[1127:1127+51, 0])))
# axarr[1].plot(np.fft.rfftfreq(50, 1/50), np.abs(np.fft.rfft(data_2[527:527+51, 0])))
# axarr[1].plot(np.fft.rfftfreq(50, 1/50), np.abs(np.fft.rfft(data_2[927:927+51, 0])))


figure.show()
plt.pause(900000)
