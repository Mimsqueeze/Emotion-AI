# Emotion-AI
Have you ever experienced the frustration that comes with an insensitive, typical emotionless response from platforms like ChatGPT? To solve this, we created an *incwedible* web-app called EmotionAI, which can detect what emotions you're feeling through analyzing your facial expressions, and responding accordingly!

![image](https://github.com/Mimsqueeze/Bitcamp-2024/assets/101283845/abe85296-47f8-4124-bd5c-3b6fa03a7492)

# Our Process
## Computer Vision
We first focused on training a machine learning model to accurately detect sentiment based on faces. We initially used the FER2013 dataset but switched to AffectNet because of better resolution. We used a Convolutional Neural Network inspired by [susantabiswas](https://github.com/susantabiswas/realtime-facial-emotion-analyzer/blob/master/training/facial%20Emotions.ipynb), modifying it to work for our dataset and purposes. A major problem we faced was the model struggling to classify images in real time, so we also used OpenCV's object-detection capacities to isolate faces from an image. In the end, after hours of preprocessing, image manipulation, training, and model optimization, our models reached an average of ~94% for binary classification between "happy" and "sad" faces!

![image](https://github.com/Mimsqueeze/Bitcamp-2024/assets/101283845/296e7a21-04c2-4ded-8e8c-63d353d25eb5)

## ChatGPT Integration
Then, we connected our trained CNN with the ChatGPT API to seamlessly curate personalized responses. We achieved this by obtaining a secure API key to access the GPT API remotely and gathering predictions from our CNN to generate modified prompts to feed into GPT. The idea we were going for is if the model detected you were feeling "happy", then ChatGPT would generate "happy" and "cutesy" responses, while if you were feeling sad, the model's energy would match your energy, and thus produce deadpan responses. Of course, this can be modified to different types of responses and the models can be changed to detect different emotions as well.

## React + TypeScript + Flask
Finally, it was time for UI design! We used the React Framework with TypeScript, along with traditional CSS to make a very clean and simple interface allowing the user to communicate with our EmotionAI. React + TypeScript for the front end and Python Flask for the backend allowed us to connect the UI to the Machine Learning model, which output results to be used by our ChatGPT integration to generate sentiment-based responses. The UI design was inspired by Bitcamp's 10th anniversary, adding a sentimental touch. 

# Acknowledgements
This was a project made by Minsi Hu, Davy Wang, Isabelle Park, and Zewen Ye at Bitcamp 2024 held at the University of Maryland :) Special thanks to Andy Qu for guiding us on the machine-learning aspects of the project, and all of the Bitcamp Organizers for making the event possible - we had a blast! >:D
