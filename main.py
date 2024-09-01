import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf

# Load the TensorFlow model
model = tf.keras.models.load_model('path_to_your_model_directory')

# Define class names based on your model's classes
class_names = ["Happy", "Sad", "Angry"]  # Update with your classes

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = np.array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict emotion
def predict_emotion(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)
    return class_names[predicted_class[0]]

# Function to capture and display the video frame
def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    # Predict the emotion on the current frame
    emotion = predict_emotion(frame)

    # Convert the frame to RGB format and display it
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.configure(image=imgtk)
    
    # Display the detected emotion
    emotion_label.config(text=f"Detected Emotion: {emotion}")
    
    label.after(10, show_frame)

# Set up the Tkinter window
root = tk.Tk()
root.title("Emotion Detection")

# Create a label for the video feed
label = Label(root)
label.pack()

# Create a label to display the detected emotion
emotion_label = Label(root, text="Detected Emotion: ", font=("Arial", 18))
emotion_label.pack()

# Start capturing and displaying video frames
show_frame()

# Run the Tkinter main loop
root.mainloop()

# Release the webcam when the app is closed
cap.release()
