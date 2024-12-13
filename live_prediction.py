import numpy as np
import cv2
import mediapipe as mp
import time
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
import pickle
import os

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()

# Load your trained CNN model
model = load_model('trained_model.h5')
model.summary()

# Load labels
mapping_file_path = r"ResultMap.pkl"
with open(mapping_file_path, "rb") as mapping_file:
    Result_class = pickle.load(mapping_file)

# Create the labels list from the mapping
labels = [Result_class[i] for i in range(len(Result_class))]

font = cv2.FONT_HERSHEY_SIMPLEX

def start():
    # Start video capture
    cap = cv2.VideoCapture(0)
    print("Starting video stream...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a natural selfie-view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Face Detection
        results = face_detection.process(rgb_frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.detections:
            print(f"Number of faces detected: {len(results.detections)}")
            for idx, detection in enumerate(results.detections):
                mp_drawing.draw_detection(frame, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Ensure the bounding box has a valid size
                if w > 0 and h > 0:
                    try:
                        # Extract and resize the face region
                        face = frame[y:y + h, x:x + w]
                        face_img = cv2.resize(face, (100, 100))  # Resize to match model input size
                        cv2.imshow("Fata din cadru",face_img)

                        face_img  = image.img_to_array(face_img)
                        face_img = np.expand_dims(face_img, axis= 0)

                        # Make prediction
                        result = model.predict(face_img, verbose = 0)
                        prediction = Result_class[np.argmax(result)]

                        # Display prediction on the frame
                        cv2.putText(frame, prediction, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


                    except Exception as e:
                        print(f"Error processing face {idx + 1}: {e}")

        # Show the result
        cv2.imshow('Face Recognition', frame)

        # Quit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


