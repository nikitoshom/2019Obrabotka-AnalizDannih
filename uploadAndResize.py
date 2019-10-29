import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#import pathlib
#import os
import numpy as np
path = '/Users/nikitaskuratovich/Downloads/Images/Trees.jpg'
rates = (2, 4, 8, 16, 32)
img = tf.keras.preprocessing.image.load_img(path)
img = tf.convert_to_tensor(np.asarray(img))
size = img.shape
images = []

print(size[0], size[1])
for rate in rates:
    X=int(size[0]/rate)
    Y=int(size[1]/rate)
    print(X,'  ',Y)
    image = tf.image.resize(img, (X,Y))
    print(image.shape)
    images[rate] = image
    to_img = tf.keras.preprocessing.image.array_to_img(image)
    to_img.show()

