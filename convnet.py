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

train_db = pd.read_csv("data/emnist-bymerge-train.csv",header=None)
test = pd.read_csv("data/emnist-bymerge-test.csv",header=None)
y_train = train_db.iloc[:,0]
y_train = np_utils.to_categorical(y_train, 47)
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

print ("x_train:",x_train.shape)
	
model = Sequential()
model.add(Conv2D(32, 3, 3, border_mode='valid', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation("relu"))
model.add(Conv2D(32, 3, 3,  border_mode='valid'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(47))
model.add(BatchNormalization())
model.add(Activation("softmax"))

print model.summary()


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=128,
          epochs=70,
          verbose=1)
model.save('output/Model.h5', overwrite=True)
model.save_weights('output/Weights.h5',overwrite=True)
'''
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''




