import cv2
import mediapipe as mp
import os
from datetime import datetime
import time 

 # Thresholds for regions in the bounding box
EDGE_THRESHOLD_LEFT_RIGHT = 0.15  # Threshold for left/right edges
CENTER_THRESHOLD = 0.2            # Threshold for the center region

# Counters to track saved frames for each region
save_counts = {
    "center": 0,
    "left": 0,
    "right": 0,
    "up": 0,
    "down": 0
}


class AddFace:
    def __init__(self, max_faces_region, parent_folder='detect_faces', confidence_threshold=0.5):
        self.parent_folder = parent_folder
        self.max_faces = max_faces_region * 3
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        self.saved_faces = []
        self.max_faces_region = max_faces_region
        self.save_folder = self.create_save_folder()
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection()
        self.mp_drawing = mp.solutions.drawing_utils

        os.makedirs(self.parent_folder, exist_ok=True)

    def create_save_folder(self):
        folder_name = input("Enter the desired folder name: ")
        save_folder = os.path.join(self.parent_folder, folder_name)
        os.makedirs(save_folder, exist_ok=True)
        print(f"Saving images to folder: {save_folder}")
        return save_folder

    def process_frame(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face_detection.process(img_rgb)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                # Get bounding box and nose keypoint
                bbox = detection.location_data.relative_bounding_box
                keypoints = detection.location_data.relative_keypoints
                nose_point = keypoints[0]  # Nose is the first keypoint

                # Bounding box dimensions
                h, w, _ = img.shape
                xmin, ymin = int(bbox.xmin * w), int(bbox.ymin * h)
                bbox_width = int(bbox.width * w)
                bbox_height = int(bbox.height * h)

                # Calculate relative nose position
                relative_x = (nose_point.x - bbox.xmin) / bbox.width
                relative_y = (nose_point.y - bbox.ymin) / bbox.height
                
                # Check if nose is in specific regions
                frame_name = None
                if (CENTER_THRESHOLD < relative_x < 1 - CENTER_THRESHOLD and
                        CENTER_THRESHOLD < relative_y < 1 - CENTER_THRESHOLD and
                        save_counts["center"] < self.max_faces_region):
                    frame_name = f"center_{save_counts['center']}.jpg"
                    save_counts["center"] += 1
                elif relative_x < EDGE_THRESHOLD_LEFT_RIGHT and save_counts["left"] < self.max_faces_region:
                    frame_name = f"left_{save_counts['left']}.jpg"
                    save_counts["left"] += 1
                elif relative_x > 0.7 - EDGE_THRESHOLD_LEFT_RIGHT and save_counts["right"] < self.max_faces_region:
                    frame_name = f"right_{save_counts['right']}.jpg"
                    save_counts["right"] += 1

            # Save the frame if a condition is satisfied
            if frame_name:
                cropped_face = img_bgr[ymin:ymin+bbox_height, xmin:xmin+bbox_width]
                face_path = os.path.join(self.save_folder, frame_name)
                cv2.imwrite(face_path, cropped_face)
                print(f"Frame saved: {frame_name}")

        return img_bgr

    def run(self):
        webcam = cv2.VideoCapture(0)

        while webcam.isOpened():
            success, img = webcam.read()
            if not success:
                break
            
            img_processed = self.process_frame(img)

            # Stop if max faces saved
            if len(os.listdir(self.save_folder)) >= self.max_faces:
                print(f"Folder '{self.save_folder}' contains more than {self.max_faces} face images. Exiting...")
                break

            cv2.imshow("Face Detection", img_processed)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

        webcam.release()
        cv2.destroyAllWindows()


