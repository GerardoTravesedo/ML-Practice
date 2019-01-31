from multiprocessing import Pool
from os import listdir
from os import getpid
import dataset_generator
import pickle
import math


TRAINING_INPUT_FOLDER = "../dataset-training-test/training/"

# TODO: We have to generate ROIS in a different way for test dataset
TEST_INPUT_FOLDER = "../dataset-training-test/test/"

TRAINING_IMAGE_FOLDER = TRAINING_INPUT_FOLDER + "/image/"
TRAINING_ANNOTATION_FOLDER = TRAINING_INPUT_FOLDER + "/annotation/"

TRAINING_OUTPUT_FOLDER = "../dataset-rcnn/training/"

NUMBER_THREADS = 2


def task(paths, output_folder):
    """
    Subprocess task that finds the rcnn input data for each pair (image, annotation) and writes
    it into a file in pickle format
    """
    # Generate the rcnn input data for each image
    data = [dataset_generator.get_image_data_training(path[0], path[1]) for path in paths]

    output_file = "{}rcnn_dataset_{}".format(output_folder, getpid())

    # Write the entire list of image data into a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Thread {} is done. Number of images processed: {}".format(getpid(), len(data)))


def main(image_folder, annotation_folder, output_folder):
    """
    Creates a specific number of threads defined by NUMBER_THREADS to generate the rcnn input data
    from the original PASCAL VOC images and annotations

    :param
        image_folder: folder that contains all the images it will generate rois for
        annotation_folder: folder that contains all the annotations for the images
        output_folder: output folder where the report will be generated
    """
    # Creating pool of subprocesses
    pool = Pool(processes=NUMBER_THREADS)

    # We get the number of images to calculate how many will be processed by each thread
    images = listdir(image_folder)
    number_images = len(images)
    images_per_thread = math.ceil(number_images / float(NUMBER_THREADS))
    print("Each thread will process a max of {} images".format(images_per_thread))

    # Keeping track of the images that are grouped together so far. Once its length gets to
    # images_per_thread we can submit the task
    images_annotations_group = []

    # Generate rcnn input data for each combination of image and annotation
    for file_pair in zip(images, listdir(annotation_folder)):
        # Finding paths to image and annotation
        image_path = image_folder + file_pair[0]
        annotation_path = annotation_folder + file_pair[1]
        # Adding them to the current group as a tuple (image_path, annotation_path)
        images_annotations_group.append((image_path, annotation_path))
        # If current group is complete
        if len(images_annotations_group) == images_per_thread:
            print("Submitting task with {} images".format(images_per_thread))
            # Submit task to thread
            pool.apply_async(task, (images_annotations_group, output_folder))
            # Reinitialize the group of paths
            images_annotations_group = []

    # Submitting task for last group in case it didn't get to images_per_thread
    if len(images_annotations_group) != 0:
        print("Submitting task with {} images".format(len(images_annotations_group)))
        pool.apply_async(task, (images_annotations_group, output_folder))

    print("Done submitting all tasks to threads")

    pool.close()
    pool.join()

if __name__ == '__main__':
    training_image_folder = "../test/data/test-batch-reader-dataset/images/"
    training_annotation_folder = "../test/data/test-batch-reader-dataset/annotations/"
    training_output_folder = "../test/data/test-batch-reader-dataset/"
    main(training_image_folder, training_annotation_folder, training_output_folder)
