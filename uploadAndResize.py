import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
#import pathlib
#import os
import numpy as np


if __name__ == '__main__':

    path = '/Users/nikitaskuratovich/Downloads/Images/Trees.jpg'
    rates = (2, 4, 8, 16, 32)
    img = tf.keras.preprocessing.image.load_img(path)
    img = tf.convert_to_tensor(np.asarray(img))
    size = img.shape
    images_resized = []

    print(size[0], size[1])
    for rate in rates:
        X=int(size[0]/rate)
        Y=int(size[1]/rate)
        #print(X,'  ',Y)
        image = tf.image.resize(img, (X,Y))
        #print(image.shape)
        images_resized.append(image)
        to_img = tf.keras.preprocessing.image.array_to_img(image)
        #to_img.show()

    # Read images from file.
    #im1 = tf.decode_png('path/to/im1.png')
    #im2 = tf.decode_png('path/to/im2.png')
    # Compute PSNR over tf.uint8 Tensors.
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

    '''
    # Compute PSNR over tf.float32 Tensors.
    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)
    psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
    # psnr1 and psnr2 both have type tf.float32 and are almost equal.
    '''
