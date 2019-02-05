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

        # Checking ground truth information
        expected_gt_data = np.array([
            np.array([{"class": "person", "bbox": np.array([214, 121, 216, 300])}]),
            np.array([{"class": "aeroplane", "bbox": np.array([124, 166, 325, 224])},
             {"class": "aeroplane", "bbox": np.array([159, 187,  76,  74])},
             {"class": "person", "bbox": np.array([234, 384, 21, 104])},
             {"class": "person", "bbox": np.array([31, 403, 21, 104])}])
        ])
        assert batch1["gt_objects"].shape == (2,)
        np.testing.assert_equal(expected_gt_data[0][0], batch1["gt_objects"][0][0])
        assert batch1["gt_objects"][0].shape == (1,)
        np.testing.assert_equal(expected_gt_data[1][0], batch1["gt_objects"][1][0])
        np.testing.assert_equal(expected_gt_data[1][1], batch1["gt_objects"][1][1])
        np.testing.assert_equal(expected_gt_data[1][2], batch1["gt_objects"][1][2])
        np.testing.assert_equal(expected_gt_data[1][3], batch1["gt_objects"][1][3])
        assert batch1["gt_objects"][1].shape == (4,)

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

