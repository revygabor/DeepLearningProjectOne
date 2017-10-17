import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rotated = "A://Machine Learning//project1//vector_lenght_test//rot.csv"
data = pd.read_csv(rotated, header= None)  # elforgatott fájl beolvasása
acceleration_rot = data.as_matrix(columns=[1, 2, 3])

figure, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(np.fft.rfftfreq(50, 1/50), np.abs(np.fft.rfft(acceleration_rot[1127:1127+51, 0])))
axarr[1].plot(np.fft.rfftfreq(50, 1/50), np.abs(np.fft.rfft(acceleration_rot[527:527+51, 0])))
axarr[2].plot(np.fft.rfftfreq(50, 1/50), np.abs(np.fft.rfft(acceleration_rot[927:927+51, 0])))

figure.show()
plt.pause(900000)
