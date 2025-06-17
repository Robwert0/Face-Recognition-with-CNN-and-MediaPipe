📸 Real-Time Face Recognition with CNN and MediaPipe

This project implements a complete, real-time face recognition system using Python. It leverages the high-performance MediaPipe library for initial face detection and a custom-trained Convolutional Neural Network (CNN) built with TensorFlow/Keras for the actual recognition task.

The entire workflow—from collecting face data to training the model and running the final recognition—is managed through a single, easy-to-use command-line interface.

📚 Table of Contents
Features

How It Works

Technology Stack

Project Structure

Setup and Installation

Usage

Contributing

License

Acknowledgements

✨ Features
Real-Time Detection & Recognition: Identifies faces from a live webcam feed.

Easy Data Collection: An interactive script to capture and label face images for new individuals.

Custom CNN Model: A robust Convolutional Neural Network that you train on your own collected data.

All-in-One Script: A single main.py script with a menu to run all functionalities.

High-Performance Detection: Utilizes Google's MediaPipe for fast and accurate initial face localization.

⚙️ How It Works
The system operates in a three-stage pipeline:

Data Collection: The script captures 200 images of a person's face from the webcam, automatically detects the face in each frame using MediaPipe, and saves the cropped face images into a labeled folder inside the Data/ directory.

Model Training: The training script loads all the collected face images, preprocesses them (resizes, converts to grayscale, normalizes), and then trains a CNN to learn the features of each individual. The trained model is saved as face_recognition_model.h5.

Face Recognition: The final script uses the trained model to perform real-time recognition. For each frame from the webcam, it:

Detects faces using MediaPipe.

Crops and preprocesses each detected face.

Feeds the face into the trained CNN to get a prediction.

Draws a bounding box and the predicted name on the video feed.

💻 Technology Stack
Python: Core programming language.

OpenCV: For camera handling and image processing.

MediaPipe: For high-performance face detection.

TensorFlow / Keras: For building and training the CNN model.

NumPy: For numerical operations on image data.

Scikit-learn: For splitting the dataset into training and testing sets.

📂 Project Structure

.
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

🚀 Setup and Installation
Prerequisites
Python 3.8+

A webcam connected to your computer.

Installation Steps
Clone the repository:

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Create a virtual environment (recommended):

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install the required libraries:
A requirements.txt file is recommended for managing dependencies. Create one with the following content:

opencv-python
mediapipe
tensorflow
numpy
scikit-learn

Then, run the following command to install them all:

pip install -r requirements.txt

▶️ Usage
The project is controlled via a simple menu. Run the main.py script from your terminal to get started.

python main.py

You will be presented with the following menu:

--- Face Recognition System Menu ---
1. Collect Face Data
2. Train the Model
3. Run Face Recognition
4. Exit

Step 1: 👤 Collect Face Data
Choose option 1.

The system will ask for the name of the person you are adding.

A window will open with your webcam feed. Hold your face steady in front of the camera. The script will automatically detect your face and save 200 images.

Repeat this process for every person you want the system to recognize.

Step 2: 🧠 Train the Model
After collecting data for at least one person, choose option 2.

The script will load the images from the Data/ directory, process them, and train the CNN model.

This may take a few minutes depending on your computer's hardware.

Once finished, face_recognition_model.h5 and labels.npy will be saved in your project directory.

Step 3: 🔍 Run Face Recognition
With a trained model, choose option 3.

A window with your webcam feed will open.

The system will now detect and recognize any person it has been trained on, displaying their name above their face.

Press q to close the window and return to the menu.

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
This project is distributed under the MIT License. See LICENSE for more information.

🙏 Acknowledgements
A special thanks to the teams behind these incredible open-source tools that made this project possible:

OpenCV

Google MediaPipe

TensorFlow
