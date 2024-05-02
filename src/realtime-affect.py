# import the opencv library 
import cv2
import numpy as np
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
from keras.models import load_model

def main():
    # define a video capture object 
    vid = cv2.VideoCapture(0) 

    emotions = ['HAPPY', 'SAD']

    dir = f"./model/affect/"
    model = load_model(f'{dir}model.h5')
    model.load_weights(f'{dir}weights.h5')

    # text on frame
    text = ""

    # frame counter
    n = 0

    # happy counter
    n_happy = 0

    # sad counter
    n_sad = 0

    while(True): 
        
        # Capture the video frame by frame 
        ret, frame = vid.read() 
    
        # Get the dimensions of the original image
        frame_matrix = np.array(frame)
        # print(frame_matrix.shape)
        # print(frame_matrix)
        height, width, _ = frame_matrix.shape

        # Calculate the size of the square (assume you want to crop a square with size min(height, width))
        size = min(height, width)

        # Calculate the starting position for cropping
        start_x = (width - size)
        start_y = (height - size)

        # Crop the middle square
        cropped_image = frame_matrix[start_y:start_y+size, start_x:start_x+size]

        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # cv2.imwrite('frame.jpg', gray_image) 
        
        haar_cascade_face_detector = cv2.CascadeClassifier('./src/haar_face.xml')

        #returns the rectangular coordinates of the face 
        faces_rect = haar_cascade_face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

        print(faces_rect)

        cropped_face = np.array([])
        #prints the number of faces detected 
        for (x,y,w,h) in faces_rect:
            cropped_face = gray_image[y:y+h, x:x+w]
            cv2.rectangle(cropped_image, (x,y), (x+w,y+h), (0,255,0), thickness=2)
            
        if (cropped_face.any()):
            # cv2.imwrite('frame.jpg', cropped_face) 

            # Resize the image to 96x96
            resized_cropped_face = cv2.resize(cropped_face, (96, 96))
            
            input_image = np.reshape(resized_cropped_face, (1, 96, 96, 1))

            # Make predictions
            predictions = model.predict(input_image)

            # Assuming it's a classification task with multiple classes, you may want to get the class with the highest probability
            predicted_class = np.argmax(predictions)

            print(predicted_class)

            # Print the predicted class
            print("Predicted class:", emotions[predicted_class])

            if (predicted_class == 0):
                n_happy += 1
            else:
                n_sad += 1

            # cv2.imwrite('saved_image.jpg', resized_cropped_face)

        text_frame = cropped_image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # White color in BGR format
        thickness = 2

        # Get the size of the text
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate the position to put the text (bottom-left corner)
        text_x = (cropped_image.shape[1] - text_size[0]) // 2
        text_y = cropped_image.shape[0] - 10

        # Put the text on the image
        cv2.putText(text_frame, text, (text_x, text_y), font, font_scale, font_color, thickness)

        # Display the resulting frame 
        cv2.imshow('frame', cropped_image) 
        
        # Update text only when n == 30
        if (n == 5):
            if (n_happy > n_sad):
                text = emotions[0]
            else:
                text = emotions[1]

            # Reset counts
            n = 0
            n_happy = 0
            n_sad = 0
                
        # Increment frame counter
        n += 1

        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # After the loop release the cap object 
    vid.release() 

    # Destroy all the windows 
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    main()

