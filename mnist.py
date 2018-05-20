import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D, Input, BatchNormalization, Activation
from keras import optimizers
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print x_train[1].shape
print len(x_test)
num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
x_train = x_train.reshape(x_train.shape[0], 28, 28,1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28,1).astype('float32')

x_train /= 255
x_test /=255
model5 = Sequential()
model5.add(Conv2D(32, 3, 3, border_mode='valid', input_shape=(28, 28, 1)))
model5.add(BatchNormalization())
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Activation("relu"))
model5.add(Conv2D(32, 3, 3,  border_mode='valid'))
model5.add(Dropout(0.25))
model5.add(BatchNormalization())
model5.add(Activation("relu"))
model5.add(MaxPooling2D(pool_size=(2, 2)))
model5.add(Dropout(0.3))
model5.add(Flatten())
model5.add(Dense(1024, activation='relu'))
model5.add(BatchNormalization())
model5.add(Dropout(0.4))
model5.add(Dense(128, activation='relu'))
model5.add(BatchNormalization())
model5.add(Dropout(0.4))
model5.add(Dense(num_classes))
model5.add(BatchNormalization())
model5.add(Activation("softmax"))
print model5.summary()
# model5 the model
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
model5.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model5_info = model5.fit(x_train, y_train, batch_size=128, \
                         nb_epoch=50, verbose=1, validation_split=0.2)

model5.save_weights('output/Weights_mnist.h5',overwrite=True)
result = model5.predict(x_test)
predicted_class = np.argmax(result, axis=1)
true_class = np.argmax(y_test, axis=1)
num_correct = np.sum(predicted_class == true_class) 
accuracy = float(num_correct)/result.shape[0]
print (accuracy * 100)


