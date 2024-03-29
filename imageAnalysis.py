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

#values
rates = (2, 4, 8, 16, 32)

# create folder for data
def CreateFolderForData (NameFolderTFRecords = 'tfrecords'):
    if not os.path.exists(NameFolderTFRecords):
        os.mkdir(NameFolderTFRecords)

# Upload images from  folder
def UploadImagesFromHolder (NameFolderWithImages = 'images'):
    uploadImages = list()
    extensions = ('jpg', 'png')
    for extension in extensions:
        images = glob.glob(os.path.join(NameFolderWithImages, '*.' + extension))
        uploadImages.extend(images)
    return uploadImages

# resize images (first - original, second-sitxh resized with original values of size )
def ResizeImages(uploadImages):
    images_after_resized_with_original_size = []
    for number_image in range(len(uploadImages)):
        img = tf.keras.preprocessing.image.load_img(uploadImages[number_image])
        img = tf.convert_to_tensor(np.asarray(img))
        images_after_resized_with_original_size.append(img)
        for rate in rates:
            size = img.shape
            X = int(size[0] / rate)
            Y = int(size[1] / rate)
            img_res = tf.image.resize(img, (X, Y))
            img_res = tf.image.resize(img_res, (size[0], size[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            img_res = tf.keras.preprocessing.image.array_to_img(img_res)
            img_res = np.asanyarray(img_res)
            images_after_resized_with_original_size.append(img_res)
    return images_after_resized_with_original_size

#PSNR values just for first image for testing and plotting
def PSNRForPlotting(images_after_resized_with_original_size):
    psnr = []
    for i in range(1, 6):
        psnrNew = tf.image.psnr(images_after_resized_with_original_size[0], images_after_resized_with_original_size[i],
                                max_val=255)
        psnr.append(psnrNew.numpy())
    return psnr

#Plot PSNR - rate
def PlotPSNR(psnr):
    ratesNP = np.asanyarray(rates)
    #valueForPlotting = [ratesNP, psnr1]
    PSNR = {'rate': ratesNP, 'PSNR': psnr}
    DFPSNR = pd.DataFrame(PSNR)
    DFPSNR.boxplot(column='PSNR', by='rate')
    plt.ylabel('PSNR')
    plt.xlabel('Compression rate')
    plt.show()

if __name__ == '__main__':
    PlotPSNR(PSNRForPlotting(ResizeImages(UploadImagesFromHolder('images'))))
