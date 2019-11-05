import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import cv2

if __name__ == '__main__':

    # create folder for data
    NameFolderTFRecords = 'tfrecords'
    if not os.path.exists(NameFolderTFRecords):
        os.mkdir(NameFolderTFRecords)

    # Upload images from  folder
    NameFolderWithImages = 'images'
    uploadImages = list()
    extensions = ('jpg', 'png')
    for extension in extensions:
        images = glob.glob(os.path.join(NameFolderWithImages, '*.' + extension))
        uploadImages.extend(images)

    # resize images (first - original, second-sitxh resized with original values of size )
    rates = (2, 4, 8, 16, 32)
    images_after_resized_with_original_size = []
    for number_image in range(len(uploadImages)):
        img = tf.keras.preprocessing.image.load_img(uploadImages[number_image])
        img = tf.convert_to_tensor(np.asarray(img))
        images_after_resized_with_original_size.append(img)
        for rate in rates:
            size = img.shape
            number = 0
            X = int(size[0] / rate)
            Y = int(size[1] / rate)
            img_res = tf.image.resize(img, (X, Y))
            img_res = tf.image.resize(img_res, (size[0], size[1]),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            img_res = tf.keras.preprocessing.image.array_to_img(img_res)
            img_res = np.asanyarray(img_res)
            images_after_resized_with_original_size.append(img_res)

    #PSNR values just for first image for testing and plotting
    psnr1 = []
    for i in range(1, 6):
        psnrNew = tf.image.psnr(images_after_resized_with_original_size[0],images_after_resized_with_original_size[i], max_val=255)
        psnr1.append(psnrNew.numpy())
    print(psnr1)

    #Plot PSNR - rate
    rates = np.asanyarray(rates)
    valueForPlotting = [rates, psnr1]
    PSNR = {'rate': rates,'PSNR': psnr1}
    DFPSNR = pd.DataFrame(PSNR)
    DFPSNR.boxplot(column='PSNR', by='rate')
    plt.ylabel('PSNR')
    plt.xlabel('Compression rate')
    plt.show() 
