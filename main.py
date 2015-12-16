import h5py
import numpy as np
import caffe
print("start")

#
SIZE = 38 # fixed size to all images
# with open( 'train.txt', 'r' ) as T :
#     lines = T.readlines()
# X = np.zeros( (len(lines), 1, 1, SIZE), dtype='f4' )
# y = np.zeros( (len(lines), 15), dtype='f4' )
# for i,l in enumerate(lines):
#     sp = l.split('_')
#     x_variables = []
#     for j,c in enumerate(sp[0]):
#         x_variables.append(float(ord(c))/255)
#     X[i] = x_variables
#
#
#     y_variables = []
#     outcomes = sp[1].split(" ")
#     for (k,item_k) in enumerate(outcomes):
#        y_variables.append(float(str(item_k).split("/n")[0])/500)
#     y[i] = y_variables
# with h5py.File('train.h5','w') as H:
#     H.create_dataset( 'X', data=X ) # note the name X given to the dataset!
#     H.create_dataset( 'y', data=y ) # note the name y given to the dataset!
# with open('train_h5_list.txt','w') as L:
#     L.write( 'train.h5' ) # list all h5 files you are going to use


X_IN = np.zeros( (1,1,1,SIZE), dtype='f4' )
input_str = "000000829,000487320,0001,000487321,000_1.80 1.9 0 0 0 0 0 0 0 0 0 0 0 0 0";

sp = input_str.split('_')
x_variables = []
for j,c in enumerate(sp[0]):
    x_variables.append(float(ord(c))/255)
X_IN  = x_variables

print(X_IN)
caffe.set_mode_gpu();
net = caffe.Classifier("trainedModel/network.prototxt","trainedModel/first_iter_80000.caffemodel")
scores = net.predict(np.array(X_IN))

print(scores)