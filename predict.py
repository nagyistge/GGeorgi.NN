import caffe
import numpy as np

SIZE = 131

X_IN = np.zeros( (1,1,1,SIZE), dtype='f4' )

input_str = "04140,000000529,000531687,000,005,001,000,000,000,000,000,000,000,000,000,000531688,000,004,000,000,000,000,000,000,000,000,000,000";

x_variables = []
for j,c in enumerate(input_str):
    x_variables.append(float(ord(c))/255)

X_IN  = x_variables

print(X_IN)

caffe.set_mode_gpu();

net = caffe.Classifier("trainedModel/network.prototxt","trainedModel/first_iter_300000.caffemodel")

scores = net.predict(np.array(X_IN))

print(scores)