import numpy as np
import pandas as pd

from numpy.fft import fft

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint


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


load0 = False
load1 = False
loading_model = True

window_size = 50
input_size = 6*window_size

# import, preprocess datas

if load0:
    inp0 = np.loadtxt("input0.txt", delimiter=',')
else:
    data0 = pd.read_csv("data_gabor.csv")
    mtx0 = data0.as_matrix()
    inp0 = preprocess(mtx0, 5, window_size)
    # np.savetxt("input0.txt", inp0, delimiter=',') #4GB nem biztos h el kell menteni

print("data0 loaded")

if load1:
    inp1 = np.loadtxt("input.txt", delimiter=',')
else:
    data1 = pd.read_csv("data_dani.csv")
    mtx1 = data1.as_matrix()
    inp1 = preprocess(mtx1, 4, window_size)
# np.savetxt("input1.txt", inp1, delimiter=',')

print("data1 loaded")

label0 = np.zeros([np.shape(inp0)[0]])
label1 = np.ones([np.shape(inp1)[0]])

inputs = np.concatenate((inp0, inp1))  # merge data
labels = np.concatenate((label0, label1))  # merge labels

data_all = np.concatenate((inputs, labels[:, np.newaxis]), axis=1)
data_all = np.random.permutation(data_all)
inputs = data_all[:, 0:input_size]
labels = data_all[:, input_size]

# build model
if loading_model:
    model = load_model("lepes_model.h5")
else:
    model = Sequential()
    model.add(Dense(256, activation="sigmoid", input_dim=input_size))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(64, activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])

early_stopping = EarlyStopping(patience=2, min_delta=0.0001, monitor="loss")
checkpoint = ModelCheckpoint("lepes_model.h5", verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

# training model
model.fit(inputs, labels, epochs=100, callbacks=[early_stopping, checkpoint], batch_size=128)

# save model
model.save("lepes_model.h5")
