from matplotlib import pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import csv
import os
import time
import pandas as pd
import tensorflow as tf
from keras import layers, models, callbacks
from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

import cv2


emotions = ['happy', 'sad']

def main():
    print(f"Processing happy images...")

    happy_dir = f"./dataset/happy/"

    # Get the list of image file names in the directory
    image_files = os.listdir(happy_dir)

    # List to store the images
    images = []

    num = 0

    # Read each image and append it to the list
    for image_file in image_files:
        # Read the image
        image_path = os.path.join(happy_dir, image_file)
        image = cv2.imread(image_path)
        
        # Check if the image was read successfully
        if image is not None:
            images.append(image)

        if (num % 100 == 0):
            print(f"On image {num}")
        num += 1

    # Convert the list of images to a NumPy array
    images = np.array(images)

    images = np.dot(images[..., :3], [0.2989, 0.5870, 0.1140])

    images = np.expand_dims(images, axis=-1)

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Fit the ImageDataGenerator to your data
    datagen.fit(images)

    # Generate augmented images
    augmented_images = []
    for batch in datagen.flow(images, batch_size=32):
        augmented_images.append(batch)
        if len(augmented_images) * 32 >= images.shape[0]:
            break

    # Convert augmented images back to numpy array
    augmented_images = np.concatenate(augmented_images, axis=0)

    # Update console
    print(f"Saving numpy array to file!")
    np.save(f'./dataset/happy.npy', augmented_images)

    print(f"Processing sad images...")
    sad_dir = f"./dataset/sad/"

    # Get the list of image file names in the directory
    image_files = os.listdir(sad_dir)

    # List to store the images
    images = []

    num = 0

    # Read each image and append it to the list
    for image_file in image_files:
        # Read the image
        image_path = os.path.join(sad_dir, image_file)
        image = cv2.imread(image_path)
        
        # Check if the image was read successfully
        if image is not None:
            images.append(image)

        if (num % 100 == 0):
            print(f"On image {num}")
        num += 1

    # Convert the list of images to a NumPy array
    images = np.array(images)

    images = np.dot(images[..., :3], [0.2989, 0.5870, 0.1140])

    images = np.expand_dims(images, axis=-1)

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Fit the ImageDataGenerator to your data
    datagen.fit(images)

    # Generate augmented images
    augmented_images = []
    for batch in datagen.flow(images, batch_size=32):
        augmented_images.append(batch)
        if len(augmented_images) * 32 >= images.shape[0]:
            break

    # Convert augmented images back to numpy array
    augmented_images = np.concatenate(augmented_images, axis=0)

    # Update console
    print(f"Saving numpy array to file!")
    np.save(f'./dataset/sad.npy', augmented_images)

    # print(images_array.shape)

    print(f"Processing neutral images...")
    neutral_dir = f"./dataset/neutral/"

    # Get the list of image file names in the directory
    image_files = os.listdir(neutral_dir)

    # List to store the images
    images = []

    num = 0

    # Read each image and append it to the list
    for image_file in image_files:
        # Read the image
        image_path = os.path.join(neutral_dir, image_file)
        image = cv2.imread(image_path)
        
        # Check if the image was read successfully
        if image is not None:
            images.append(image)

        if (num % 100 == 0):
            print(f"On image {num}")
        num += 1

    # Convert the list of images to a NumPy array
    images = np.array(images)

    images = np.dot(images[..., :3], [0.2989, 0.5870, 0.1140])

    images = np.expand_dims(images, axis=-1)

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Fit the ImageDataGenerator to your data
    datagen.fit(images)

    # Generate augmented images
    augmented_images = []
    for batch in datagen.flow(images, batch_size=32):
        augmented_images.append(batch)
        if len(augmented_images) * 32 >= images.shape[0]:
            break

    # Convert augmented images back to numpy array
    augmented_images = np.concatenate(augmented_images, axis=0)

    # Update console
    print(f"Saving numpy array to file!")
    np.save(f'./dataset/neutral.npy', augmented_images)

    print(images.shape)

if __name__ == '__main__':
    main()

