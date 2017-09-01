import numpy as np
import pandas as pd

from numpy.fft import fft

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


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
loading_model = False

window_size = 50
step = 5
input_size = 6*window_size
dropout_rate = 0.2

# importing, preprocessing data

#data0
if load0:
    inp0 = np.loadtxt("input0.txt", delimiter=',')
else:
    data0 = pd.read_csv("data_gabor.csv")
    mtx0 = data0.as_matrix()
    inp0 = preprocess(mtx0, step, window_size)
    # np.savetxt("input0.txt", inp0, delimiter=',') #4GB nem biztos h el kell menteni

print("data0 loaded")

#data1
if load1:
    inp1 = np.loadtxt("input.txt", delimiter=',')
else:
    data1 = pd.read_csv("data_dani.csv")
    mtx1 = data1.as_matrix()
    inp1 = preprocess(mtx1, step, window_size)
    # np.savetxt("input1.txt", inp1, delimiter=',')

print("data1 loaded")

label0 = np.zeros([np.shape(inp0)[0]])
label1 = np.ones([np.shape(inp1)[0]])

inputs = np.concatenate((inp0, inp1))  # merge data
labels = np.concatenate((label0, label1))  # merge labels


#permutating
permutation_array = np.arange(inputs.shape[0]) #ezzel keverjük össze az adatokat
permutation_array = np.random.permutation(permutation_array) #összekeverjük

inputs = inputs[permutation_array] #indexeljük
labels = labels[permutation_array]


# data_all = np.concatenate((inputs, labels[:, np.newaxis]), axis=1)
# data_all = np.random.permutation(data_all)
# inputs = data_all[:, 0:input_size]
# labels = data_all[:, input_size]

#train-validation data split
inputs_train, inputs_val, labels_train, labels_val = train_test_split(inputs, labels, test_size=0.33)

# building model
if loading_model:
    model = load_model("lepes_model.h5")
else:
    model = Sequential()
    model.add(Dense(256, activation="sigmoid", input_dim=input_size))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation="sigmoid"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['acc'])

early_stopping = EarlyStopping(patience=25, min_delta=0.0001, monitor="val_loss")
checkpoint = ModelCheckpoint("lepes_model.h5", verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

# training model
model.fit(inputs_train, labels_train, epochs=100, callbacks=[early_stopping, checkpoint],
          batch_size=128, verbose=1, validation_data=(inputs_val, labels_val))

#confusion matrix
confusion_mtx_train = confusion_matrix(labels_train, model.predict(inputs_train))
print("confusion matrix (train):", confusion_mtx_train)

confusion_mtx_val = confusion_matrix(labels_val, model.predict(inputs_val))
print("confusion matrix (val):", confusion_mtx_val)


# saving model
model.save("lepes_model.h5")
