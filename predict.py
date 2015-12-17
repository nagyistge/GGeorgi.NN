import caffe
import numpy as np


SIZE = 132
X_IN = np.zeros( (1,1,1,SIZE), dtype='f4' )
input_str = "04140,000000529,000531687,000,005,001,000,000,000,000,000,000,000,000,000,000531688,000,004,000,000,000,000,000,000,000,000,000,000_1 1.14 4.35 26 0 0 0 0 0 0 0 0 0 0 0 0";

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