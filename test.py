from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D, Input, Conv3D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing import image
import keras
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import keras.backend as K

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
K.set_learning_phase(False)

MODEL_NAME = "char"
def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


def model():
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
			 activation='relu',
			 input_shape=(28, 28, 1)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(47, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
		      optimizer=keras.optimizers.Adadelta(),
		      metrics=['accuracy'])
	model.load_weights("output/Weights.h5")
	print model.summary
	return model


def testing(model):
	dataframe = pd.read_csv("Labels.csv", header=None)
	dictionary = {}
	dataframe = np.asarray(dataframe)
	
	for i in range(len(dataframe[:,0])):
		dictionary[dataframe[i,0]] = dataframe[i,1]
	
	img = cv2.imread('roi.jpg',0)
	img = cv2.resize(img, (28, 28))
	cv2.imwrite("new.jpg", img)

	x = image.img_to_array( img)
	x = x.reshape(1, 28, 28, 1).T.astype('float32')
	x /= 255
	out = model.predict(x)
	
	print dictionary[np.argmax(out)]

if __name__=="__main__":
	model = model()
	print(model.summary)
	print(model.input)
	print(model.output)
	export_model(tf.train.Saver(), model, ["conv2d_1_input"], "dense_2/Softmax")
