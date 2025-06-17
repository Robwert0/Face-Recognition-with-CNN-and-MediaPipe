import cv2
import mediapipe as mp
import os
from datetime import datetime
import time 

# Thresholds for regions in the bounding box
EDGE_THRESHOLD_LEFT_RIGHT = 0.15  # Threshold for left/right edges
CENTER_THRESHOLD = 0.2            # Threshold for the center region
CLOSE_FACE_AREA_THRESHOLD = 65000  # Threshold for face proximity based on bounding box area

# Counters to track saved frames for each region
save_counts = {
    "center": 0,
}

class AddFace:
    def __init__(self, max_faces_region, parent_folder='detect_faces', confidence_threshold=0.5):
        self.parent_folder = parent_folder
        self.max_faces = max_faces_region
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
    
    def is_blurry(self, image, threshold=140.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Laplacian variance: {laplacian_var}")
        return laplacian_var < threshold
    
    def is_well_lit(self, image, brightness_threshold=(100, 220)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        print(f"Mean brightness: {mean_brightness}")
        return brightness_threshold[0] < mean_brightness < brightness_threshold[1]

    def process_frame(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face_detection.process(img_rgb)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                keypoints = detection.location_data.relative_keypoints
                nose_point = keypoints[0]  # Nose is the first keypoint

                h, w, _ = img.shape
                xmin, ymin = int(bbox.xmin * w), int(bbox.ymin * h)
                bbox_width = int(bbox.width * w)
                bbox_height = int(bbox.height * h)

                # Calculate the area of the bounding box
                bbox_area = bbox_width * bbox_height

                relative_x = (nose_point.x - bbox.xmin) / bbox.width
                relative_y = (nose_point.y - bbox.ymin) / bbox.height

                print(bbox_area)
                # Check if the face is close based on the bounding box area
                if bbox_area > CLOSE_FACE_AREA_THRESHOLD:
                    cv2.putText(img_bgr, "Face is close", (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    

                    if (CENTER_THRESHOLD < relative_x < 1 - CENTER_THRESHOLD and
                            CENTER_THRESHOLD < relative_y < 1 - CENTER_THRESHOLD and
                            save_counts["center"] < self.max_faces_region):
                        print("Center region detected")
                        frame_name = f"center_{save_counts['center']}.jpg"
                        save_counts["center"] += 1

                        cropped_face = img_bgr[ymin:ymin + bbox_height, xmin:xmin + bbox_width]

                        print(self.is_blurry(cropped_face), self.is_well_lit(cropped_face))
                        if not self.is_blurry(cropped_face) and self.is_well_lit(cropped_face):
                            face_path = os.path.join(self.save_folder, frame_name)
                            cv2.imwrite(face_path, cropped_face)
                            print(f"Frame saved: {frame_name}")
                        else:
                            print("Skipped blurry frame.")

                else:
                    cv2.putText(img_bgr, "Face is to far away", (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return img_bgr

    def run(self):
        webcam = cv2.VideoCapture(0)

        while webcam.isOpened():
            success, img = webcam.read()
            if not success:
                break
            
            img_processed = self.process_frame(img)

            # Stop if max faces saved
            print(f"Faces saved in folder: {len(os.listdir(self.save_folder))}")
            print(f"Max faces allowed: {self.max_faces}")
            if len(os.listdir(self.save_folder)) >= self.max_faces:
                print(f"Folder '{self.save_folder}' contains more than {self.max_faces} face images. Exiting...")
                break

            cv2.imshow("Face Detection", img_processed)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

        webcam.release()
        cv2.destroyAllWindows()

