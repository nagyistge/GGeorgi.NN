import caffe
import numpy as np

SIZE = 131

X_IN = np.zeros( (1,1,1,SIZE), dtype='f4' )

input_str = "04620,000000529,000531653,000,003,002,000,000,000,000,000,000,000,000,000,000531654,000,002,002,000,000,000,000,000,000,000,000,000";

for j,c in enumerate(input_str):
    X_IN[0][0][0][j] = float(ord(c))/255

caffe.set_mode_gpu();

net = caffe.Classifier("trainedModel/network.prototxt","trainedModel/first_iter_20000.caffemodel")

scores = net.predict(np.array(X_IN))

print(scores)