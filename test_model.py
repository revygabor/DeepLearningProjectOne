from keras.models import load_model

from numpy.fft import fft

import numpy as np
import pandas as pd

# def preprocess(mtx):
#     n_rows = np.shape(mtx)[0] - 49
#     inp = np.zeros([n_rows, 150])
#     for i in range(0, n_rows, 5):
#         x = mtx[i:i + 50, 0]
#         y = mtx[i:i + 50, 1]
#         z = mtx[i:i + 50, 2]
#
#         x = fft(x)
#         y = fft(y)
#         z = fft(z)
#
#         inp[i, :] = np.concatenate((x, y, z))
#
#         if i%10000 == 0:
#             print(i)
#
#     return inp

def preprocess(mtx, step, window_size):
    n_rows = np.shape(mtx)[0] - window_size + 1
    inp = np.zeros([n_rows, 6*window_size])
    for i in range(0, n_rows, step):
        x = mtx[i:i + window_size, 0]
        y = mtx[i:i + window_size, 1]
        z = mtx[i:i + window_size, 2]

        x = fft(x)
        y = fft(y)
        z = fft(z)

        inp[i, :] = np.concatenate((np.real(x), np.imag(x), np.real(y), np.imag(y), np.real(z), np.imag(z)))

        if i % 10000 == 0:
            print(i)

    return inp

# loading data
data = pd.read_csv("A:/Machine Learning/project1/data/teszt/converted.txt")
mtx = data.as_matrix()
inp = preprocess(mtx, 5, 50)

# predicting
model = load_model("lepes_model.h5")
pred = model.predict(inp)
print(np.array_str(pred))
