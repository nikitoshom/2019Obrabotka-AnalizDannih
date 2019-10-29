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
'''
downsampled_im = tf.io.decode_jpeg(tf.reshape(img, []))[..., :3]
original_im = tf.io.decode_jpeg(img)#[..., :3]
size = original_im.shape[0:2]
psnr1 = tf.image.psnr(images_resized[0], img, max_val=255)
resized = tf.image.resize(
        downsampled_im,
        size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        preserve_aspect_ratio=False,
        antialias=False,
        name=None
    )
psnr_value = tf.image.psnr(
        original_im,
        resized,
        max_val=255,
        name=None)
'''
