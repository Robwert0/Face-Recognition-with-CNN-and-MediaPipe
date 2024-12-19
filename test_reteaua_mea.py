import image_processing
import glob
import os
from keras._tf_keras.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Paths
train_images = r"train_images"
test_images = r"test_images"
test_data_path = r"test_data"
model_save_path = r"trained_model.keras"

# Step 1: Create data generators
training_data, testing_data = image_processing.create_generators(train_images, test_images)

# Step 2: Create class mapping
result_class = image_processing.create_class_mapping(training_data)

# Step 3: Number of output neurons based on class mapping
output_neurons = len(result_class)
print('\nNumber of output neurons: ', output_neurons)

# Step 4: Create or Load the Model
if os.path.exists(model_save_path):
    print("Loading pre-trained model...")
    model = image_processing.load_model(model_save_path)
else:
    print("Training a new model...")
    model = image_processing.create_model(output_neurons)
    image_processing.train_model(model, training_data, testing_data)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}.")

# Step 5: Prediction Logic
def predict_images(model, img_folder, class_mapping):
    """Predict classes for all images in a given folder."""
    img_paths = glob.glob(os.path.join(img_folder, "*.jpg"))
    if not img_paths:
        print("No images found in the specified folder.")
        return

    print(f"Found {len(img_paths)} images. Predicting...")
    
    for path in img_paths:
        try:
            # Load and preprocess the image
            test_image = image.load_img(path, target_size=(100, 100))
            test_image = image.img_to_array(test_image) / 255.0
            test_image = np.expand_dims(test_image, axis=0)

            # Make prediction
            result = model.predict(test_image, verbose=0)
            probabilities = {class_mapping[idx]: prob for idx, prob in enumerate(result[0])}
            predicted_class = max(probabilities, key=probabilities.get)
            confidence = probabilities[predicted_class]

            # Display image and result
            plt.imshow(image.load_img(path))
            plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}")
            plt.axis('off')
            plt.show()

            print(f'Prediction: {predicted_class}, Confidence: {confidence:.2f}, All Probabilities: {probabilities}')
        except Exception as e:
            print(f"Error processing {path}: {e}")

# Predict images from test data
predict_images(model, test_data_path, result_class)
