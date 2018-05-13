import numpy as np
import csv
import matplotlib.pyplot as plt

import pandas as pd
	
from keras.utils import np_utils
train_db = pd.read_csv("emnist-balanced-train.csv",header=None)
test = pd.read_csv("emnist-balanced-test.csv",header=None)
y_train = train_db.iloc[:,0]
#y_train = np_utils.to_categorical(y_train, 47)
y_test = test.iloc[:,0]

print ("y_train:", y_train.shape)
	

x_train = train_db.iloc[:,1:]
x_test = test.iloc[:,1:]
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
x_train = x_train.reshape(x_train.shape[0], 28, 28,1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28,1).astype('float32')
x_train /= 255
x_test /=255

print x_train[1]


j = 35

for i in range(len(x_train)):
	label = y_train[i]
	
	if j == label:
		j+=1
		print label
		
		plt.imshow(x_train[i].reshape(28,28).T, cmap = "gray_r")
		plt.show()
	
	if j==47:
		break
