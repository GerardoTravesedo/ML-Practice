import cv2
import numpy as np
import pickle


def reshape_image(image):
    """
    This function converts the original shape 1x3072 from CIFAR-10 into shape 32x32x3 BGR
    that we can show using cv2

    :param
        image: pixels representing the image. It is an array with 3072 entries:
        The first 1024 entries contain the red channel values, the next 1024 the green,
        and the final 1024 the blue
    """
    # If the image is a 1D array convert it into (1024, 3)
    if image.shape == (1, 3072) or image.shape == (3072,):
        # Splitting by color RGB
        image = image.reshape(3, 1024)
        # Converting image into BGR for cv2 and turning rows into columns
        image = np.array([image[2], image[1], image[0]]).transpose()
        # Putting image into 32x32x3
        image = image.reshape(32, 32, 3)

    return image


def show_image_from_pixels(image):
    """
    :param
        image: pixels representing the image (BGR).
    """
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize_image(image, height, width):
    """
    Given an image of size (1 x 3072), it first converts it into shape (32 x 32 x 3) and then
    it resizes it to (height x width x 3)

    :param
        image: raw CIFAR-10 image with shape 1 x 3072.
        height: new height for the image
        width: new width for the image
    """
    image = reshape_image(image)
    image = cv2.resize(image, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    return image


def scale_images_cifar10_file(original_file, destination_file, height, width):
    """
    Resizes all images in the original CIFAR-10 pickle files into shape (224 x 224 x 3) and
    stores them back into pickle format

    :param
        original_file: CIFAR-10 original file
        destination_file: new file with resized images in pickle format
        height: new height for the image
        width: new width for the image
    """
    with open(original_file, 'rb') as input_file:
        # Shape (10000, 3072)
        data = pickle.load(input_file)
        # Shape (10000, 224, 224, 3)
        data["data"] = np.apply_along_axis(resize_image, 1, data["data"], height, width)

    with open(destination_file, 'wb') as output_file:
        pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    print "Done converting file " + original_file


def reshape_images_cifar10_file(original_file, destination_file):
    """
    Reshapes all images in the original CIFAR-10 pickle files into shape (32 x 32 x 3) and
    stores them back into pickle format

    :param
        original_file: CIFAR-10 original file
        destination_file: new file with resized images in pickle format
    """
    with open(original_file, 'rb') as input_file:
        # Shape (10000, 3072)
        data = pickle.load(input_file)
        # Shape (10000, 224, 224, 3)
        data["data"] = np.apply_along_axis(reshape_image, 1, data["data"])

    with open(destination_file, 'wb') as output_file:
        pickle.dump(data, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    print "Done converting file " + original_file
