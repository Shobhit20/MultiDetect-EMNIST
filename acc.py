import numpy as np
import pandas as pd

from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D, Input, Conv3D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import pandas as pd
import keras
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import test as testing

test = pd.read_csv("emnist-balanced-test.csv",header=None)

x_test = test.iloc[:,1:]
x_test = np.asarray(x_test)
model = testing.model()
model.load_weights("output/Weights.h5")

y_test = test.iloc[:,0]
print np.asarray(y_test)
arr = []	
correct = 0
for i in range(len(y_test)):
	x = x_test[i]
	x = x.reshape(1, 28, 28, 1).astype('float32')
	x /= 255
	model.predict(x)
	out = model.predict(x)
	arr.append(np.argmax(out))
	print i
	if arr[i] == y_test[i]:
		correct+=1

print correct
print correct/(len(y_test))





