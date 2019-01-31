import ImageTools
import DatasetReader

input_directory = "./dataset/"

output_directory = "./dataset-reshaped/"

list_batch_file = \
  ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]

# for cifar10_file in list_batch_file:
#     ImageTools.reshape_images_cifar10_file(
#         input_directory + cifar10_file, output_directory + cifar10_file)


reader = DatasetReader.DatasetReader("./dataset-reshaped/data_batch_1", 3)
batch = reader.get_batch()
print batch["labels"][1]
ImageTools.show_image_from_pixels(batch["data"][1])

