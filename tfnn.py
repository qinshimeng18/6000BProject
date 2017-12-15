from __future__ import absolute_import, division, print_function

import tflearn
import numpy as np
from loadData import train_x, train_y, test_x, test_y

data = train_x
label = train_y

input_layer = tflearn.input_data(shape=[None, 100])
network = tflearn.fully_connected(input_layer, 50, activation='relu', weights_init='xavier', regularizer='L1')
network = tflearn.fully_connected(network, 10, activation='relu', weights_init='xavier', regularizer='L1')
output = tflearn.fully_connected(network, 1)

net = tflearn.regression(output, loss='mean_square', learning_rate=0.00001)

model = tflearn.DNN(net)
model.fit(data, label, n_epoch=100, shuffle=True)

prediction = model.predict(test_x)

prediction = np.round(prediction)

print('Test accuracy:', np.mean(prediction == test_y))