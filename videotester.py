import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import load_model
import matplotlib.pyplot as plt
import keyboard  # Install with 'pip install keyboard' if you don't have it

# Load model
model = load_model("best_model.keras")

# Load Haar cascade for face detection
face_haar_cascade = cv2.CascadeClassifier("C:/Users/Joarder/AppData/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()  # Captures frame and returns boolean value and captured image
    if not ret:
        continue
    
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # Cropping region of interest i.e., face area from image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # Find max indexed array
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Resize and convert to RGB for displaying with Matplotlib
    resized_img = cv2.resize(test_img, (1000, 700))
    plt.imshow(resized_img)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()  # Clear the plot for the next frame

    # Break the loop if 'q' is pressed
    if keyboard.is_pressed('q'):
        break

cap.release()
plt.close()  # Close the plot window
cv2.destroyAllWindows()
