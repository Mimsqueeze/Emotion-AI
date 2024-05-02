from flask import Flask
import cv2
import numpy as np
import numpy as np
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# define a video capture object 
vid = cv2.VideoCapture(0) 
dir = f"./model/affect/"
model = load_model(f'{dir}model.h5')
model.load_weights(f'{dir}weights.h5')
haar_cascade_face_detector = cv2.CascadeClassifier('./haar_face.xml')

@app.route('/api/ml')
def predict():

    emotions = ['HAPPY', 'SAD']

    # frame counter
    n = 0

    # happy counter
    n_happy = 0

    # sad counter
    n_sad = 0

    while(n < 8): 
        
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
                
        # Increment frame counter
        n += 1

    if (n_happy > n_sad):
        print("FINAL VERDICT: HAPPY")
        return {'sentiment': 0}
    else:
        print("FINAL VERDICT: SAD")
        return {'sentiment': 1}