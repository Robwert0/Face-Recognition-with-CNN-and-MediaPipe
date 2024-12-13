from add_face_to_database import AddFace
import image_processing

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
    train_images = r"detect_faces"
    training_data, testing_data = image_processing.create_generators(train_images)

    # Create class mapping
    result_class = image_processing.create_class_mapping(training_data, testing_data)

    # Number of output neurons based on class mapping
    output_neurons = len(result_class)
    print("\nNumber of output neurons: ", output_neurons)

    # Create the CNN model
    model = image_processing.create_model(output_neurons)
    model.save("trained_model.h5")

    # Train the model
    image_processing.train_model(model, training_data, testing_data)

    import live_prediction
    # Start live prediction
    live_prediction.start()

if __name__ == "__main__":
    main()
