from picamera import PiCamera
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow.keras as keras
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import layers, models


camera = PiCamera()
camera.resolution = (150,150)
camera.rotation = 180
camera.start_preview(fullscreen=False, window=(30,30,320,240))
for i in range(1,4):
    print(4-i)
    sleep(1)
camera.capture('/home/pi/Transfer_learning/imagen.jpg')
camera.stop_preview()
camera.close()

from tensorflow.keras import layers, models

model=keras.models.load_model('Modelo_Vgg16.h5')

img_path = '/home/pi/Transfer_learning/imagen.jpg'
img = image.load_img(img_path, target_size=(150,150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print(model.predict(x))
plt.imshow(img)
