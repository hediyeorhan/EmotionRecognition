import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import scipy
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2


train_dir = 'train'
test_dir = 'test'

train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(directory=train_dir,target_size=(48,48),class_mode='categorical', color_mode = "grayscale", batch_size=64)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = train_datagen.flow_from_directory(directory=test_dir,target_size=(48,48),class_mode='categorical', color_mode = "grayscale",batch_size=64)


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, (3, 3), activation = "relu", input_shape = (48, 48, 1)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.25),
                                    tf.keras.layers.Conv2D(16, (3, 3), activation = "relu"),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.25),
                                    tf.keras.layers.Conv2D(32, (3, 3), activation = "relu"),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Dropout(0.25),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(7, activation = "softmax")])

model.summary()

model.compile(optimizer = RMSprop(lr=0.001), loss = "categorical_crossentropy", metrics=["accuracy"])

class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('accuracy') > 0.80):
          self.model.stop_training = True

callbacks = myCallback()
history = model.fit(train_generator, epochs=500, validation_data = test_generator, callbacks = [callbacks])


plt.figure(figsize=(24,8))

plt.subplot(1,2,1)
plt.plot(history.history["val_accuracy"], label="validation_accuracy", linewidth=4)
plt.plot(history.history["accuracy"], label="training_accuracy", linewidth=4)
plt.legend()
plt.title("ACC" ,fontsize=18)

plt.subplot(1,2,2)
plt.plot(history.history["val_loss"], label="validation_loss", color="red", linewidth=4)
plt.plot(history.history["loss"], label="training_loss", color="cyan", linewidth=4)
plt.legend()

plt.title("LOSS" ,fontsize=18)

plt.show()


face_clsfr = cv2.CascadeClassifier("C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")


labels_dict={0:'ANGRY', 1:'DISGUST', 2:'FEAR', 3:'HAPPY', 4:'NEUTRAL', 5:'SAD', 6:'SURPRISE'}
color_dict={0:(0, 0, 0)}

rect_size = 11
cap = cv2.VideoCapture(0) 


while True:
    (ret, img) = cap.read()
    
    img = cv2.flip(img, 1, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    resized = cv2.resize(img,(48,48))
    faces = face_clsfr.detectMultiScale(resized, 1.3, 5) 
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f] 
     
        face_img = img[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(48, 48))
        normalized=resized/255.0
        reshaped = np.reshape(normalized, (1,48,48,1))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[0],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[0],-1) # -40
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) # -10

    cv2.imshow('EMOTION_DETECTION',   img)
    key = cv2.waitKey(10)
    
    if key == 27:   # Esc
        break

cap.release()

cv2.destroyAllWindows()