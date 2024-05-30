# import the libraries
import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import datetime
import cv2
import numpy as np
import os
import csv
import datetime
import tensorflow as tf
import face_recognition

# Load face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load student images and names for recognition
path = 'student_images'
studentImg = []
studentName = []
myList = os.listdir(path)
for cl in myList:
    curimg = cv2.imread(f'{path}/{cl}')
    studentImg.append(curimg)
    studentName.append(os.path.splitext(cl)[0])

# Function to find face encodings for recognition
def findEncoding(images):
    imgEncodings = []
    for img in images:
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_recognition.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings

# Get face encodings for student images
EncodeList = findEncoding(studentImg)

# Load emotion detection model
model = tf.keras.models.load_model('facial_expression_model.h5')  # Load your emotion detection model here

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Open CSV file for writing and write headers
with open('students_data.csv', mode='a', newline='') as file:  # Append mode
    writer = csv.writer(file)
    # Check if the file is empty
    if file.tell() == 0:
        writer.writerow(['Name', 'Expression', 'Date', 'Time'])  # Write headers

    # Start video capture
    vid = cv2.VideoCapture(0)
    while True:
        success, frame = vid.read()
        if not success:
            break

        # Get current date and time
        current_date = datetime.datetime.now().date()
        current_time = datetime.datetime.now().strftime('%H:%M:%S')

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y:y + h, x:x + w]

            # Resize to match model input size for emotion detection
            face_roi_resized = cv2.resize(face_roi, (48, 48))
            face_roi_resized = np.expand_dims(face_roi_resized, axis=0)
            face_roi_resized = np.expand_dims(face_roi_resized, axis=-1)

            # Predict emotion
            predicted_emotion = model.predict(face_roi_resized)
            emotion_label = emotion_labels[np.argmax(predicted_emotion)]

            # Recognize student and add to CSV
            face_gray = gray[y:y+h, x:x+w]
            face_rgb = frame[y:y+h, x:x+w]
            Smaller_frame = cv2.resize(face_rgb, (0, 0), None, 0.25, 0.25)

            facesInFrame = face_rec.face_locations(Smaller_frame)
            encodeFacesInFrame = face_rec.face_encodings(Smaller_frame, facesInFrame)

            for encodeFace, faceLoc in zip(encodeFacesInFrame, facesInFrame):
                matches = face_rec.compare_faces(EncodeList, encodeFace)
                faceDis = face_rec.face_distance(EncodeList, encodeFace)
                print(faceDis)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = studentName[matchIndex].upper()
                    writer.writerow([name, emotion_label, current_date, current_time])

                    # Draw rectangle around the face and label the emotion and name
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f'{name}: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Facial Expression and Person Identification', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Flush changes to the CSV file to ensure auto-saving
        file.flush()

    # Release the video capture
    vid.release()
