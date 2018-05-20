import numpy as np
import pandas as pd

from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D, Input, BatchNormalization, Activation
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

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model_info = model.fit(x_train, y_train, batch_size=128, \
                         nb_epoch=50, verbose=1, validation_split=0.2)

model.save('output/Model.h5', overwrite=True)
model.save_weights('output/Weights.h5',overwrite=True)
result = model5.predict(x_test)
predicted_class = np.argmax(result, axis=1)
true_class = np.argmax(y_test, axis=1)
num_correct = np.sum(predicted_class == true_class) 
accuracy = float(num_correct)/result.shape[0]
print (accuracy * 100)




