import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

# MNIST DATA SET
mnist = tf.keras.datasets.mnist

# dividing the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalising
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# creating a neural network
model = tf.keras.models.Sequential()

# fully connected layer
#befor connected layer , need to be flaaten so that 2D to 1D
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) 

# connected layer 1
model.add(tf.keras.layers.Dense(128, activation='relu')) # Activation - remove <0; allow >0

# connected layer 2
model.add(tf.keras.layers.Dense(128, activation='relu'))

# last connected layer
# last dense layer must be 10
# last activation layer must be softmax
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#compiling model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#training model
model.fit(x_train, y_train, epochs=3)

# evaluating on testing dataset MNIST
loss,accuracy = model.evaluate(x_test ,y_test)
print ("total loss on 10000 test samples",loss)
print("validation accuracy on 10000 test sample",accuracy)

#save model
model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"this digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("error!")
    finally: 
        image_number += 1
    import tensorflow as tf
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train.shape
#just check the graph, how data looks like
import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show() #in order to execute the graph
plt.imshow(x_train[0], cmap=plt.cm.binary) #colored img to binary image
print(x_train[0])
## image value(0 to 255) , normalise them(x_train/255)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
plt.imshow(x_train[0], cmap = plt.cm.binary)
print(x_train[0])
print(y_train[0])
import numpy as np
IMG_SIZE= 28
x_trainr= np.array(x_train).reshape(-1,IMG_SIZE, IMG_SIZE,1) #increasing one dimension for kernel operation
x_testr= np.array(x_test).reshape(-1,IMG_SIZE, IMG_SIZE,1) #increasing one dimension for kernel operation
print("training sample dimension",x_trainr.shape)
print("testing sample dimension",x_testr.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
#creating a neural network
model=Sequential()

#first convolution layer  (60000,28,28,1) 28-3+1=26*26
model.add(Conv2D(64,(3,3), input_shape = x_trainr.shape[1:])) #only for first convolution layer to mention input layer size
model.add(Activation("relu")) #activation function to make it non-linear, <0 remove;>0 allow
model.add(MaxPooling2D(pool_size=(2,2))) #maxpooling single maximumvalue of 2x2

#2nd convolution layer  
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#3rd convolution layer
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#fully connected layer #1
model.add(Flatten()) #before using fully connected layer, need to be flatten so that 2D to 1D
model.add(Dense(64))
model.add(Activation("relu"))

#fully connected layer #2
model.add(Flatten())
model.add(Dense(32))
model.add(Activation("relu"))

#fully connected layer #1
model.add(Flatten())
model.add(Dense(10)) #classes is 10
model.add(Activation('softmax'))
model.summary()
print("total Training Sample",len(x_trainr))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_trainr, y_train, epochs=5) #training model
# evaluating on testing dataset MNIST
loss,accuracy = model.evaluate(x_testr,y_test)
print ("total loss on 10000 test samples",loss)
print("validation accuracy on 10000 test sample",accuracy)
predictions = model.predict([x_testr])
print (predictions) #these are only array containing softmax probabilities
print(np.argmax(predictions[0])) #to convert the prediction we use numpy,for that
plt.imshow(x_test[0])