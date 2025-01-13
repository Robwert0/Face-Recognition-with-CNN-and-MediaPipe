import os
import shutil
import random

# Path to your main folder
main_folder = 'detect_faces'

# Subfolders for each class
classes = [d for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]

# Directory to store the new train/test folders
train_dir = 'train_images'
test_dir = 'test_images'

# Create the train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create subfolders for each class inside train_images and test_images
for class_name in classes:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

# Function to move images into train and test directories
def move_images(class_name, source_folder, train_folder, test_folder, split_ratio=0.8):
    class_folder = os.path.join(source_folder, class_name)
    
    # Get all image files in the class folder
    images = [f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
    
    # Shuffle images to ensure random split
    random.shuffle(images)
    
    # Split images into train and test sets
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]
    
    # Move train images to train folder
    for img in train_images:
        shutil.move(os.path.join(class_folder, img), os.path.join(train_folder, class_name, img))

    # Move test images to test folder
    for img in test_images:
        shutil.move(os.path.join(class_folder, img), os.path.join(test_folder, class_name, img))

# Organize the images for each class
for class_name in classes:
    move_images(class_name, main_folder, train_dir, test_dir)

print("Images have been successfully organized into train and test directories.")
