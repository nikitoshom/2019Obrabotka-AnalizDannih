import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import cv2
import os
import glob

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
    original_images=[]
    img = []
    images_resized = []
    images_after_resized_with_original_size = []
    number_image = 0
    for number_image in range(len(uploadImages)):
        img = tf.keras.preprocessing.image.load_img(uploadImages[number_image])
        img = tf.convert_to_tensor(np.asarray(img))
        original_images.append(img)
        for rate in rates:
            size = img.shape
            number = 0
            X = int(size[0] / rate)
            Y = int(size[1] / rate)
            # print(X,'  ',Y) #checking result
            image = tf.image.resize(img, (X, Y))
            # print(image.shape) #checking result
            images_resized.append(image)  # add new risized image
            image_original_size = tf.image.resize(image, (size[0], size[1]),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            images_after_resized_with_original_size.append(image_original_size)
            number = number + 1

            to_img = tf.keras.preprocessing.image.array_to_img(image_original_size)  # array to image
            #to_img.show() #checking result
    print(len(images_after_resized_with_original_size))

    to_img = tf.keras.preprocessing.image.array_to_img(images_after_resized_with_original_size[4])
    to_img.show()
    for i in range(0,5):
        # image = np.asanyarray(image)

        psnr1 = tf.image.psnr(np.asarray(original_images[0]), np.asarray(images_after_resized_with_original_size[i]), max_val=255)
        print(psnr1)


    '''
    print(len(images_resized))
    # decode to image with using methode nearest neighbor
    size = img.shape
    resized = tf.image.resize(
        images_resized[1],
        size,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        preserve_aspect_ratio=False,
        antialias=False,
        name=None
    )
    '''

    '''
    psnr_value = tf.image.psnr(
        original_im,
        resized,
        max_val=255,
        name=None)
    '''

# to_img.show() #checking result
#list of resized images
#print(size) #checking result
#print(size[0], size[1]) #checking result





    #to_img2 = tf.io.encode_jpeg(image,size[0],size[1])

    # Read images from file.
    #im1 = tf.decode_png('path/to/im1.png')
    #im2 = tf.decode_png('path/to/im2.png')
    # Compute PSNR over tf.uint8 Tensors.
    #downsampled_im = tf.io.decode_jpeg(tf.reshape(img, []))[..., :3]


    '''
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
