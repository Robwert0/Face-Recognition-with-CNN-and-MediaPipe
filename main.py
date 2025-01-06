from add_face_to_database import AddFace
import image_processing
import os

model_save_path = r"trained_model.keras"
train_images = r"train_images"
test_images = r"test_images"

def main():
    while True:
        cond_add = input("Do you want to add a new face to train the model? Y/N \n").strip().upper()
        if cond_add == "Y":
            add_face = AddFace(max_faces_region=400)
            add_face.run()
        elif cond_add == "N":
            break
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")

    # Proceed with training if at least one face was added
    training_data, testing_data = image_processing.create_generators(train_images, test_images)

    # Create class mapping
    result_class = image_processing.create_class_mapping(training_data)

    # Number of output neurons based on class mapping
    output_neurons = len(result_class)
    print("\nNumber of output neurons: ", output_neurons)

    # Create or Load the Model
    if os.path.exists(model_save_path):
        print("Loading pre-trained model...")
        model = image_processing.load_model(model_save_path)
    else:
        print("Training a new model...")
        model = image_processing.create_model(output_neurons)
        image_processing.train_model(model, training_data, testing_data)
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}.")

    import live_prediction
    # Start live prediction
    live_prediction.start()

if __name__ == "__main__":
    main()
