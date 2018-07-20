from PIL import Image
import numpy as np
import os
import glob
import re
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

def read_img(location):
	path = os.path.abspath('.cnn.py') #path
	path = re.sub('[a-zA-Z\s._]+$', '', path) #remueve basura
	x_train = []
	y_train = []
	x_test = []
	y_test = []
	dirs = os.listdir(path+'UCMerced_LandUse/Images/')
	label = 0
	for i in dirs:
		n = 0
		count = 0
		for pic in glob.glob(path+'UCMerced_LandUse/Images/'+i+'/*.tif'):
			im = Image.open(pic)
			im = np.array(im)
			if((im.shape[0]==256) and (im.shape[1] ==256) and count <90): #90/d
				r = im[:,:,0]
				g = im[:,:,1]
				b = im[:,:,2]
				if(n<5): # 5 datos para test
					x_test.append([r,g,b])
					y_test.append([label])
				else: #rdatos a entrenar
					x_train.append([r,g,b])
					y_train.append([label])
				n = n + 1
				count = count + 1
		#print(count)
		label = label + 1
	return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)

img_rows = 256
img_cols = 256
num_class = 21
x_train,y_train,x_test,y_test = read_img('UCMerced_LandUse/Images/')

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, 21)
y_test = keras.utils.to_categorical(y_test, 21)
'''print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(y_test[0:10])
print(y_train)'''

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(21, activation='softmax'))
model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=50, nb_epoch=100, verbose=1, validation_data=(x_test, y_test))
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

#############################################################

# serializa model en JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# carga json y crea model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluaa loaded model en test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))