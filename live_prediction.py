import numpy as np
import cv2
import mediapipe as mp
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
import pickle
import os
import image_processing
from datetime import datetime
from filterpy.kalman import KalmanFilter

model_save_path = r"trained_model.keras"

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
face_detection = mp_face_detection.FaceDetection()

# Initialize the Kalman filter
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])  # Transition matrix
kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Measurement matrix
kf.P *= 1000  # Initial uncertainty
kf.R = np.array([[5, 0], [0, 5]])  # Measurement noise

# Load the Model
if os.path.exists(model_save_path):
    print("Loading pre-trained model...")
    model = image_processing.load_model(model_save_path)

# Load labels
mapping_file_path = r"ResultMap.pkl"
with open(mapping_file_path, "rb") as mapping_file:
    Result_class = pickle.load(mapping_file)

# Dictionary to track detected faces
face_tracking = {}  # {name: {'confidences': [...], 'frames_absent': 0}}
logged_names = set()  # Keep track of names already logged

# Function to log presence in a text file
def log_presence(name, confidence):
    """Log detected person's name, date, time, and confidence score to a text file."""
    if name in logged_names:
        return  # Skip logging if the name is already in the log

    file_path = "prezenta.txt"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, "a") as file:
        file.write(f"{name},{current_time},{confidence:.2f}%\n")
    print(f"✅ Logged: {name} at {current_time} with confidence: {confidence:.2f}%")

    logged_names.add(name)  # Mark name as logged

def smooth_bbox(x, y, w, h):
    """Applies Kalman Filter to stabilize bounding box coordinates."""
    global kf

    # Predict step
    kf.predict()

    # Update step
    z = np.array([x + w / 2, y + h / 2])  # Measurement
    kf.update(z)

    # Extract updated values
    x_new, y_new = kf.x[0] - w / 2, kf.x[1] - h / 2
    return int(x_new), int(y_new), w, h

def normalize_lighting(face):
    """Applies histogram equalization to improve contrast."""
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_eq = cv2.equalizeHist(face_gray)
    return cv2.cvtColor(face_eq, cv2.COLOR_GRAY2BGR)

def sharpen_image(face):
    """Applies a sharpening filter to enhance details."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(face, -1, kernel)

def align_face(face, frame, landmarks):
    """Aligns the face based on eye landmarks."""
    left_eye = np.array([landmarks[33].x * frame.shape[1], landmarks[33].y * frame.shape[0]])
    right_eye = np.array([landmarks[263].x * frame.shape[1], landmarks[263].y * frame.shape[0]])

    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    M = cv2.getRotationMatrix2D(tuple(left_eye), angle, 1)
    aligned_face = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))
    return aligned_face

def exponential_moving_average(values, alpha=0.2):
    """Applies EMA to smooth confidence scores."""
    smoothed_values = []
    ema = values[0]  # Start with first value
    for val in values:
        ema = alpha * val + (1 - alpha) * ema
        smoothed_values.append(ema)
    return smoothed_values[-1]


def start():
    # Start video capture
    cap = cv2.VideoCapture(0)
    print("Starting video stream...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        detected_faces = set()  # Track names detected in this frame

        if results.detections:
            for idx, detection in enumerate(results.detections):
                mp_drawing.draw_detection(frame, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)


                if w > 0 and h > 0:
                    try:
                        # Smooth bounding box
                        x, y, w, h = smooth_bbox(x, y, w, h)

                        # Extract and resize the face region
                        face = frame[y:y + h, x:x + w]
                        face = normalize_lighting(face)
                        face = sharpen_image(face)

                        #Align face using lnadmarks
                        landmarks_result = face_mesh.process(rgb_frame)
                        if landmarks_result.multi_face_landmarks:
                            for face_landmarks in landmarks_result.multi_face_landmarks:
                                face = align_face(face, frame, face_landmarks.landmark)
                        else:
                            print("No landmarks detected in this frame.")

                        
                        # Resize the face image
                        face_img = cv2.resize(face, (224, 224))
                        face_img = image.img_to_array(face_img)
                        face_img = np.expand_dims(face_img, axis=0)

                        # Make prediction
                        result = model.predict(face_img, verbose=0)
                        max_index = np.argmax(result)
                        prediction = Result_class[max_index]
                        confidence = result[0][max_index] * 100  # Convert to percentage

                        # Track confidence scores
                        if prediction not in face_tracking:
                            face_tracking[prediction] = {'confidences': [], 'frames_absent': 0}
                        
                        face_tracking[prediction]['confidences'].append(confidence)

                        # Keep only the last 10 frames
                        if len(face_tracking[prediction]['confidences']) > 10:
                            face_tracking[prediction]['confidences'].pop(0)

                        mean_confidence = exponential_moving_average(face_tracking[prediction]['confidences'])

                        if mean_confidence >= 60:
                            detected_faces.add(prediction)
                            text = f"{prediction}: {mean_confidence:.2f}%"
                            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    except Exception as e:
                        print(f"Error processing face {idx + 1}: {e}")

        # Check if any tracked faces have disappeared
        for name in list(face_tracking.keys()):
            if name not in detected_faces:
                face_tracking[name]['frames_absent'] += 1

                # If a face disappears for 5 frames, log it (only if confidence > 60%)
                if face_tracking[name]['frames_absent'] >= 5:
                    mean_confidence = np.mean(face_tracking[name]['confidences'])
                    if mean_confidence >= 60 and name not in logged_names:  # Only log if confidence is above 60%
                        log_presence(name, mean_confidence)
            else:
                face_tracking[name]['frames_absent'] = 0  # Reset counter if face is detected again

        # Show the result
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    face_detection.close()
    face_mesh.close()
    cv2.destroyAllWindows()

# Start the application
if __name__ == "__main__":
    start()
