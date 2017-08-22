import numpy as np
import pandas as pd

from numpy import sin, cos


def rotMtx(thetaVect):  # forgatási mátrix lértrehozása szögekből
    xMtx = np.matrix([[1, 0, 0],
                      [0, cos(-thetaVect[0]), -sin(-thetaVect[0])],
                      [0, sin(-thetaVect[0]), cos(-thetaVect[0])]])

    yMtx = np.matrix([[cos(-thetaVect[1]), 0, sin(-thetaVect[1])],
                      [0, 1, 0],
                      [-sin(-thetaVect[1]), 0, cos(-thetaVect[1])]])

    zMtx = np.matrix([[cos(-thetaVect[2]), -sin(-thetaVect[2]), 0],
                      [sin(-thetaVect[2]), cos(-thetaVect[2]), 0],
                      [0, 0, 1]])

    return [xMtx, yMtx, zMtx]


person_idx = int(input('Személy indexe: '))  # 0/1
in_path = input('Adatok elérési útja (mappa): ')  # pl. A:/Machine Learning/
out_path = input('Mentés helye (fájl): ')  # pl. A:/Machine Learning/data.txt

out = np.empty([0, 4])  # ebben egyesíti a végén az adatokat

import glob

input_list = glob.glob(in_path + '/' + '*.csv')  # listába rakja a mappában lévő .csv fájlokat

n_files = len(input_list)  # .csv fálok száma a mappában
print(n_files, 'files found')

for n in range(n_files):

    data_all = pd.read_csv(input_list[n])  # fájl beolvasása

    attitude = data_all.as_matrix(columns=['attitude_roll(radians)',
                                           'attitude_pitch(radians)',
                                           'attitude_yaw(radians)'])

    acceleration = data_all.as_matrix(columns=['user_acc_x(G)',
                                               'user_acc_y(G)',
                                               'user_acc_z(G)'])

    acc_done = np.empty([0, 3])  # ebben egyesíti az átalakított adatokat

    length = attitude.shape[0]  # mtx sorainak száma

    for i in range(0, length):
        rot_mtx = rotMtx(attitude[i])
        acc_done = np.concatenate((acc_done, acceleration[i] * rot_mtx[0] * rot_mtx[1] * rot_mtx[2]))

    acc_done = np.append(acc_done, np.full([length, 1], person_idx), axis=1)
    out = np.concatenate((out, acc_done))

    print(n + 1, '/', n_files, "done")

np.savetxt(out_path, out, delimiter=',')
print("Done")