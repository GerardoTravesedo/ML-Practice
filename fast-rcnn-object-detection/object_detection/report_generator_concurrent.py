from multiprocessing import Pool
from os import listdir
from os import getpid
import dataset_generator

INPUT_FOLDER = "../dataset-training-test/training/"

IMAGE_FOLDER = INPUT_FOLDER + "image/"
ANNOTATION_FOLDER = INPUT_FOLDER + "annotation/"

OUTPUT_FOLDER = "../reports/"


def task(image_path, annotation_path, output_folder):
    """
      Subprocess task that finds the ROIs for a given image and writes the information to the
      thread's report file
    """
    output_file = "{}roi_info_per_image_{}".format(output_folder, getpid())

    with open(output_file, 'a') as f:
        data = dataset_generator.get_image_data(image_path, annotation_path)
        foreground_rois = len(data["rois"])
        background_rois = len(data["rois_background"])
        roi_info_image = {"image": image_path, "rois": foreground_rois + background_rois,
                          "foreground": foreground_rois, "background": background_rois}
        f.write(str(roi_info_image) + "\n")

    print("Done processing image: " + image_path)


def main(image_folder, annotation_folder, output_folder):
    """
    Generates a report (file) with information about the rois for each image, including:
    - Name of the file
    - Total number of rois
    - Number of foreground rois
    - Number of background rois

    :param
        image_folder: folder that contains all the images it will generate rois for
        annotation_folder: folder that contains all the annotations for the images
        output_folder: output folder where the report will be generated
    """
    pool = Pool(processes=10)
    # Generate report info for each combination of image and annotation
    for file_pair in zip(listdir(image_folder), listdir(annotation_folder)):
        # Finding paths to image and annotation
        image_path = image_folder + file_pair[0]
        annotation_path = annotation_folder + file_pair[1]
        # Submitting tasks to process pool
        pool.apply_async(task, (image_path, annotation_path, output_folder))

    pool.close()
    pool.join()

if __name__ == '__main__':
    main(IMAGE_FOLDER, ANNOTATION_FOLDER, OUTPUT_FOLDER)
