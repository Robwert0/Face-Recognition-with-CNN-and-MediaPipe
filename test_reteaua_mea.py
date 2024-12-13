import image_processing
import glob
import os
from keras._tf_keras.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

train_images = r"train_images"
test_images = r"test_images"


# Create data generators
training_data, testing_data = image_processing.create_generators(train_images, test_images)

# Create class mapping
result_class = image_processing.create_class_mapping(training_data, testing_data)

# Number of output neurons based on class mapping
output_neurons = len(result_class)
print('\nNumber of output neurons: ', output_neurons)

# Create the CNN model
model = image_processing.create_model(output_neurons)

# Train the model
image_processing.train_model(model, training_data, testing_data)

main_ = r'test_data'
img_path = glob.glob(os.path.join(main_, "*.jpg"))

print(img_path[0:5])
print('*' * 50)

print("Class distribution in training data:", training_data.class_indices)


# Predict and print accuracy for each image
for path in img_path:
    test_image = image.load_img(path, target_size=(100, 100))  # Resize to match training size

    test_image = image.img_to_array(test_image) / 255.0       # Normalize pixel values (0-1)
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image, verbose=0)
    probabilities = {result_class[idx]: prob for idx, prob in enumerate(result[0])}

    predicted_class = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted_class]

    print(f'Prediction: {predicted_class}, Confidence: {confidence:.2f}, All Probabilities: {probabilities}')
