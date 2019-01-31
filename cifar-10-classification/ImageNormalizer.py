import numpy as np
import pickle

input_directory = "./dataset-reshaped/"

output_directory = "./dataset-reshaped-normalized/"

list_batch_file = \
  ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

list_test_file = ["test_batch"]

total_training_images = 50000


def per_pixel_normalization():
    mean_matrix = np.zeros((32, 32, 3))

    # First we traverse all the images to calculate a final mean image with shape 32 x 32 x 3 where
    # each pixel is the average of the pixel in that position from all images in training set
    for cifar10_file in list_batch_file:
        with open(input_directory + cifar10_file, 'rb') as input_file:
            data = pickle.load(input_file)["data"] / float(total_training_images)
            mean_matrix = np.sum(data, 0)

    # Now we traverse the images again to subtract the mean image from the training images to
    # normalize them. We also write them to a file.
    for cifar10_file in list_batch_file + list_test_file:
        with open(input_directory + cifar10_file, 'rb') as input_file:
            data = pickle.load(input_file)
            normalized = data["data"] - mean_matrix
            data["data"] = normalized

        with open(output_directory + cifar10_file, 'wb') as output_file:
            pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def verify_pixel_normalization(file_to_check):
    with open(input_directory + file_to_check, 'rb') as original_file:
        original_data = pickle.load(original_file)["data"][0]

    with open(output_directory + file_to_check, 'rb') as generated_file:
        generated_data = pickle.load(generated_file)["data"][0]

    print str(original_data[0][0][0])
    print str(generated_data[0][0][0])


#per_pixel_normalization()
verify_pixel_normalization("data_batch_1")
