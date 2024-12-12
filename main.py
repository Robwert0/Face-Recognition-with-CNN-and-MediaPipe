from add_face_to_database import AddFace
import image_processing


def main():
    cond_add = input('Do you want to add a new face to train the model ? Y/N \n')
    if cond_add == "Y":
        add_face = AddFace(max_faces_region = 20)
        add_face.run()

        train_images = r"detect_faces"

        #create data gnerators
        training_data, testing_data = image_processing.create_generators(train_images)

        #create class mapping 
        result_class = image_processing.create_class_mapping(training_data, testing_data)

        #Numbr of output neurons based on class maping
        output_neurons = len(result_class)
        print('\nNumber onoutput neurons: ', output_neurons)

        # Create the CNN model
        model = image_processing.create_model(output_neurons)
        model.save('trained_model.h5')

        # Train the model
        image_processing.train_model(model, training_data, testing_data)
        
    import live_prediction
    live_prediction.start()
if __name__ == "__main__":
    main()