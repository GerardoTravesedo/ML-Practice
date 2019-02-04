import object_detection.dataset_reader as dataset_reader
import numpy as np


class TestRoiTools(object):

    def test_read_mini_batches(self):
        input_folder = "test/data/test-batch-reader-dataset/"
        input_files = [input_folder + "rcnn_dataset_12276", input_folder + "rcnn_dataset_12277"]
        target = dataset_reader.DatasetReader(input_files, 2, 64, 16)

        batch1 = target.get_batch()
        assert batch1["images"].shape == (2, 600, 600, 3)
        assert batch1["rois"].shape == (2, 64, 4)
        assert batch1["class_labels"].shape == (2, 64, 21)
        # Checking that there are 59 background rois for every image
        assert np.sum(batch1["class_labels"][0], axis=0)[0] == 59
        assert np.sum(batch1["class_labels"][1], axis=0)[0] == 59

        assert batch1["reg_target_labels"].shape == (2, 64, 4)

        batch2 = target.get_batch()
        assert batch2["images"].shape == (1, 600, 600, 3)
        assert batch2["rois"].shape == (1, 64, 4)
        assert batch2["class_labels"].shape == (1, 64, 21)
        # Checking that there are 59 background rois for the image
        assert np.sum(batch1["class_labels"][0], axis=0)[0] == 59
        assert batch2["reg_target_labels"].shape == (1, 64, 4)

        batch3 = target.get_batch()
        assert batch3 == {}

