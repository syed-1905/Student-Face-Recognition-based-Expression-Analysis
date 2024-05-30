#import the libraries
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Function to load images and labels from a directory
def load_images_from_directory(directory):
    images = []
    labels = []
    label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    for label in label_map:
        label_directory = os.path.join(directory, label)
        for filename in os.listdir(label_directory):
            img_path = os.path.join(label_directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(label_map[label])
    return np.array(images), np.array(labels)

# Load images and labels from the train and test directories

train_directory = "FacialExpression/train"
test_directory = "FacialExpression/test"
X_train, y_train = load_images_from_directory(train_directory)
X_test, y_test = load_images_from_directory(test_directory)

# Split the dataset into train, validation, and test sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Preprocess the   
X_train = X_train.reshape((-1, 48, 48, 1)) / 255.0
X_val = X_val.reshape((-1, 48, 48, 1)) / 255.0
X_test = X_test.reshape((-1, 48, 48, 1)) / 255.0
y_train = to_categorical(y_train, num_classes=7)
y_val = to_categorical(y_val, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)


# Define the CNN model

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax') 
])

# Compile the model

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# cnn code stored Hierarchical Data Format version 5 
model.save("facial_expression_model.h5")

model = tf.keras.models.load_model('facial_expression_model.h5')

#resize the image 
def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)


# Function to find face encodings for recognition
def findEncoding(images):
    imgEncodings = []
    for img in images:
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_recognition.face_encodings(img)[0]  # Corrected the function name
        imgEncodings.append(encodeimg)
    return imgEncodings




