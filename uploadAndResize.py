import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
import glob
#from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #path = '/Users/nikitaskuratovich/Downloads/Images/Trees.jpg'

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
    #print(len(uploadImages))

    # resize images
    rates = (2, 4, 8, 16, 32)
    #original_images=[]
    #images_resized = []
    images_after_resized_with_original_size = []



    #resized images
    for number_image in range(len(uploadImages)):
        img = tf.keras.preprocessing.image.load_img(uploadImages[number_image])
        img = tf.convert_to_tensor(np.asarray(img))
        images_after_resized_with_original_size.append(img)
        for rate in rates:
            size = img.shape
            number = 0
            X = int(size[0] / rate)
            Y = int(size[1] / rate)
            # print(X,'  ',Y) #checking result
            img_res = tf.image.resize(img, (X, Y))
            # print(image.shape) #checking result
            #images_resized.append(img_res)  # add new risized image
            img_res = tf.image.resize(img_res, (size[0], size[1]),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            img_res = tf.keras.preprocessing.image.array_to_img(img_res)
            # to_img.show()
            img_res = np.asanyarray(img_res)
            # to_img.show()
            images_after_resized_with_original_size.append(img_res)
            #number = number + 1
            #to_img = tf.keras.preprocessing.image.array_to_img(image_original_size)  # array to image
            #to_img.show() #checking result
    #print(len(images_after_resized_with_original_size))

    psnr1 = []
    for i in range(1, 6):
        psnrNew = tf.image.psnr(images_after_resized_with_original_size[0],images_after_resized_with_original_size[i], max_val=255)
        # psnr1.append(psnrNew)
        psnr1.append(psnrNew.numpy())
    # ImagesNewSize[4].show()
    # image = np.asanyarray(image)
    print(psnr1)

    decrease2 = [2, 4, 8, 16, 32]
    aswdw = [decrease2, psnr1]
    print(aswdw)

    data = {'Name': psnr1,
            'Age': decrease2}

    df = pd.DataFrame(data)
    print(df)

    plt.boxplot(df)
    plt.show()



