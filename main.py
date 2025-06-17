from add_face_to_database import AddFace
import image_processing
import os

# File paths
model_save_path = r"trained_model.keras"
train_images = r"train_images"
test_images = r"test_images"

def main():
    while True:
        cond_add = input("Do you want to add a new face to train the model? Y/N \n").strip().upper()
        if cond_add == "Y":
            add_face = AddFace(max_faces_region=300)
            add_face.run()
        elif cond_add == "N":
            break
        else:
            print("Invalid input. Please enter 'Y' or 'N'.")

    print("\nGenerating training, validation, and testing datasets...")
    training_data, validation_data, testing_data = image_processing.create_generators(train_images, test_images)

    print("\nCreating class mapping...")
    result_class = image_processing.create_class_mapping(training_data)

    output_neurons = len(result_class)
    print("\nNumber of output neurons:", output_neurons)

    if os.path.exists(model_save_path):
        print("\nPre-trained model found.")
        retrain = input("Do you want to re-train the model with updated data? Y/N \n").strip().upper()
        if retrain == "Y":
            print("Re-training the model...")
            model, base_model = image_processing.create_model(output_neurons)
            image_processing.train_model(model, training_data, validation_data)

            fine_tune = input("Do you want to fine-tune the model for better accuracy? Y/N \n").strip().upper()
            if fine_tune == "Y":
                image_processing.fine_tune_model(model, base_model, training_data, validation_data)

            model.save(model_save_path)
            print(f"Model saved to {model_save_path}.")
        else:
            print("Loading pre-trained model...")
            model = image_processing.load_model(model_save_path)
    else:
        print("No pre-trained model found. Training a new model...")
        model, base_model = image_processing.create_model(output_neurons)
        image_processing.train_model(model, training_data, validation_data)

        fine_tune = input("Do you want to fine-tune the model for better accuracy? Y/N \n").strip().upper()
        if fine_tune == "Y":
            image_processing.fine_tune_model(model, base_model, training_data, validation_data)

        model.save(model_save_path)
        print(f"Model saved to {model_save_path}.")

    print("\nEvaluating the model on the testing data...")
    test_loss, test_accuracy = model.evaluate(testing_data)
    print(f"Testing Loss: {test_loss}")
    print(f"Testing Accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
