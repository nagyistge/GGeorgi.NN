import h5py
import numpy as np
print("start")

#
SIZE = 131
with open( 'train.txt', 'r' ) as T :
    lines = T.readlines()
X = np.zeros( (len(lines), 1, 1, SIZE), dtype='f4' )
y = np.zeros( (len(lines), 16), dtype='f4' )
for i,l in enumerate(lines):
    sp = l.split('_')
    x_variables = []
    for j,c in enumerate(sp[0]):
        x_variables.append(float(ord(c))/255)
    X[i] = x_variables

    y_variables = []
    outcomes = sp[1].split(" ")
    for (k,item_k) in enumerate(outcomes):
       y_variables.append(float(str(item_k).split("/n")[0])/500)
    y[i] = y_variables
with h5py.File('train.h5','w') as H:
    H.create_dataset( 'X', data=X )
    H.create_dataset( 'y', data=y )
with open('train_h5_list.txt','w') as L:
    L.write( 'train.h5' )


with open( 'train_test.txt', 'r' ) as T :
    lines = T.readlines()
X = np.zeros( (len(lines), 1, 1, SIZE), dtype='f4' )
y = np.zeros( (len(lines), 16), dtype='f4' )
for i,l in enumerate(lines):
    sp = l.split('_')
    x_variables = []
    for j,c in enumerate(sp[0]):
        x_variables.append(float(ord(c))/255)
    X[i] = x_variables

    y_variables = []
    outcomes = sp[1].split(" ")
    for (k,item_k) in enumerate(outcomes):
       y_variables.append(float(str(item_k).split("/n")[0])/500)
    y[i] = y_variables
with h5py.File('train_test.h5','w') as H:
    H.create_dataset( 'X', data=X )
    H.create_dataset( 'y', data=y )
with open('train_h5_list_test.txt','w') as L:
    L.write( 'train_test.h5' )


