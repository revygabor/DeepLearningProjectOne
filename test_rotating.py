import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.models.layouts import Column


def ddd_vector_length(data_mtx):
    return np.sqrt(np.square(data_mtx[:, 0]) + np.square(data_mtx[:, 1]) + np.square(data_mtx[:, 2]))

# in_path = input('Adatok elérési útja (mappa): ') # két file helye
# original = in_path + '/' + input('eredeti faljnev (+.csv): ')
# rotated = in_path + '/' + input('elforgatott faljnev (+.csv): ')

original = "A://Machine Learning//project1//vector_lenght_test//orig.csv"
rotated = "A://Machine Learning//project1//vector_lenght_test//rot.csv"

data = pd.read_csv(original)  # eredeti fájl beolvasása

acceleration_orig = data.as_matrix(columns=['user_acc_x(G)',
                                           'user_acc_y(G)',
                                           'user_acc_z(G)'])
acceleration_length_orig = ddd_vector_length(acceleration_orig)


# create a new plot
s1 = figure(width=1000, plot_height=250, title='original')
s1.line(np.arange(acceleration_length_orig.size), acceleration_length_orig)

print("original plotted")

data = pd.read_csv(rotated, header= None)  # elforgatott fájl beolvasása

acceleration_rot = data.as_matrix(columns=[1, 2, 3])
acceleration_length_rot = ddd_vector_length(acceleration_rot)


# create and another
s2 = figure(width=1000, height=250, title='rotated')
s2.line(np.arange(acceleration_length_rot.size), acceleration_length_rot, line_color='orange')

# create another one
s3 = figure(width=1000, height=250, title='difference')
diff = np.subtract(acceleration_length_orig, acceleration_length_rot)
s3.line(np.arange(acceleration_length_orig.size), diff, line_color='red')# create another one
s4 = figure(width=1000, height=250, title='difference-1')
diff2 = np.subtract(acceleration_length_orig[0], acceleration_length_rot[1:])
s3.line(np.arange(acceleration_length_orig.size), diff2, line_color='red')# create another one
s4 = figure(width=1000, height=250, title='difference+1')
diff3 = np.subtract(acceleration_length_orig, acceleration_length_rot+1)
s3.line(np.arange(acceleration_length_orig.size), diff3, line_color='red')



# put all the plots in a VBox
p = Column(s1, s2, s3)

# show the results
show(p)

print("rotated plotted")

input('Press key to end...')

