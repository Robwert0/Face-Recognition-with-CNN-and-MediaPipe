Real-Time Face Recognition with CNN and MediaPipeThis project implements a complete real-time face recognition system using Python. It leverages the high-performance MediaPipe library for initial face detection and a custom-trained Convolutional Neural Network (CNN) built with TensorFlow/Keras for the actual recognition task.The entire workflow—from collecting face data to training the model and running the final recognition—is managed through a single, easy-to-use command-line interface.FeaturesReal-Time Detection & Recognition: Identifies faces from a live webcam feed.Easy Data Collection: An interactive script to capture and label face images for new individuals.Custom CNN Model: A robust Convolutional Neural Network that you train on your own collected data.All-in-One Script: A single main.py script with a menu to run all functionalities.High-Performance Detection: Utilizes Google's MediaPipe for fast and accurate initial face localization.How It WorksThe system operates in a three-stage pipeline:Data Collection: The script captures 200 images of a person's face from the webcam, automatically detects the face in each frame using MediaPipe, and saves the cropped face images into a labeled folder inside the Data/ directory.Model Training: The training script loads all the collected face images, preprocesses them (resizes, converts to grayscale, normalizes), and then trains a CNN to learn the features of each individual. The trained model is saved as face_recognition_model.h5.Face Recognition: The final script uses the trained model to perform real-time recognition. For each frame from the webcam, it:Detects faces using MediaPipe.Crops and preprocesses each detected face.Feeds the face into the trained CNN to get a prediction.Draws a bounding box and the predicted name on the video feed.Project Structure.
├── Data/
│   ├── Person_A/
│   │   ├── Person_A_0.jpg
│   │   └── ...
│   └── Person_B/
│       ├── Person_B_0.jpg
│       └── ...
├── main.py
├── face_recognition_model.h5  (Generated after training)
├── labels.npy                 (Generated after training)
└── README.md
Setup and InstallationPrerequisitesPython 3.8+A webcam connected to your computer.Installation StepsClone the repository:git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
Create a virtual environment (recommended):# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
Install the required libraries:Create a file named requirements.txt in the project directory and add the following lines:opencv-python
mediapipe
tensorflow
numpy
scikit-learn
Then, run the following command to install them all:pip install -r requirements.txt
UsageThe project is controlled via a simple menu. Run the main.py script from your terminal to get started.python main.py
You will be presented with the following menu:--- Face Recognition System Menu ---
1. Collect Face Data
2. Train the Model
3. Run Face Recognition
4. Exit
Step 1: Collect Face DataChoose option 1.The system will ask for the name of the person you are adding.A window will open with your webcam feed. Hold your face steady in front of the camera. The script will automatically detect your face and save 200 images.Repeat this process for every person you want the system to recognize.Step 2: Train the ModelAfter collecting data for at least one person, choose option 2.The script will load the images from the Data/ directory, process them, and train the CNN model.This may take a few minutes depending on your computer's hardware.Once finished, face_recognition_model.h5 and labels.npy will be saved in your project directory.Step 3: Run Face RecognitionWith a trained model, choose option 3.A window with your webcam feed will open.The system will now detect and recognize any person it has been trained on, displaying their name above their face.Press q to close the window and return to the menu.LicenseThis project is licensed under the MIT License. See the LICENSE file for details.AcknowledgementsOpenCV for camera handling and image processing.MediaPipe for providing a powerful and efficient face detection framework.TensorFlow/Keras for the tools to build and train the neural network.
