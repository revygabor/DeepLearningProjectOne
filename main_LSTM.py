import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

def preprocess(mtx):
    # arr_of_lists = []
    # for i in range(0, mtx.shape[0] - 150, 50):
    #     list = []
    #     slice = mtx[i:i + 150, 0:3]
    #     for arr in slice:
    #         list.append(arr)#np.asarray(arr))
    #     arr_of_lists.append(list)
    # return arr_of_lists
    done = np.empty((0, 150, 3))
    for i in range(0, mtx.shape[0] - 150, 50):
        slice = mtx[i:i + 150, 0:3]
        done = np.append(done, np.expand_dims(slice, axis=0), axis=0)
    return done


# importing, preprocessing data

# data0
mtx0 = pd.read_csv("data_gabor.csv").as_matrix()
print("data0 loaded")

# data1
mtx1 = pd.read_csv("data_dani.csv").as_matrix()
print("data1 loaded")

# creating sequences
sequence0 = preprocess(mtx0)
print('sequence0 created')
sequence1 = preprocess(mtx1)
print('sequence1 created')
sequence = np.append(sequence0, sequence1, axis=0)

# creating labels
label0 = np.zeros([sequence0.shape[0]])
label1 = np.ones([sequence1.shape[0]])
labels = np.append(label0, label1)

# permutating
permutation_array = np.arange(sequence.shape[0])  # ezzel keverjük össze az adatokat
permutation_array = np.random.permutation(permutation_array)  # összekeverjük

sequence = sequence[permutation_array]  # indexeljük
labels = labels[permutation_array]

# train-validation data split
inputs_train, inputs_val, labels_train, labels_val = train_test_split(sequence, labels, test_size=0.33)

#creating model
model = Sequential()

# model.add(LSTM(
#     input_dim=3,
#     output_dim=50,
#     return_sequences=True))
model.add(LSTM(input_shape=(None, 3), units=50, return_sequences=True))
model.add(Dropout(0.2))

# model.add(Dense(
#     output_dim=1, activation='linear'))
model.add(Dense(units=1, activation="linear"))

model.compile(loss='mse', optimizer='rmsprop')

model.fit(inputs_train, labels_train, validation_data=(inputs_val, labels_val))
