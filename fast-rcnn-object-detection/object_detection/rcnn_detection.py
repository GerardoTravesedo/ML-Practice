import output_analyzer
import tensorflow as tf
import time
import learning_rate_manager as rm
import numpy as np
import dataset_reader as ds_reader
import rcnn_net
import config.config_reader as conf_reader


class RCNNDetection:

    def __init__(self, session, config):
        # Shape of this placeholder is generally (2, 600, 600, 3), but when the last batch in a
        # file is read, the batch size can be less than 2 (just one image left).
        # That's why we specify None
        # image_input_batch = tf.placeholder(
        #     tf.float32, shape=(None, CHANNEL_PIXELS, CHANNEL_PIXELS, NUMBER_CHANNELS),
        #     name="ImageInputBatch")

        self._config = config

        # We are only using one image per batch in this version
        self._image_input_batch = tf.placeholder(
            tf.float32, shape=
            (1, config.get_number_image_pixels(),
             config.get_number_image_pixels(),
             config.get_number_image_channels()),
            name="ImageInputBatch")

        # Each RoI has 4 values (x, y, h, w)
        self._roi_input_batch = tf.placeholder(
            tf.float32, shape=(None, config.get_roi_bbox_fields()),
            name="RoiInputBatch")

        self._class_label_batch = tf.placeholder(
            tf.int32, shape=(None, config.get_number_classes()),
            name="ClassLabelsBatch")

        self._detection_label_batch = tf.placeholder(
            tf.int32, shape=(None, config.get_number_regression_fields()),
            name="DetectionLabelsBatch")

        self._learning_rate = tf.placeholder(tf.float32, name="LearningRate")

        self._sess = session

    def get_net(self):
        return rcnn_net.get_net(
            self._config.get_number_classes,
            self._config.get_number_regression_fields,
            self._config.get_number_resnet_layers(),
            self._config.get_number_hidden_nodes(),
            self._image_input_batch,
            self._roi_input_batch,
            self._class_label_batch,
            self._detection_label_batch,
            self._learning_rate)

    def train_net(self, training, multitask_loss, training_batch_files):
        """
        This function trains the rcnn network

        :param training: tensorflow operator to train the network (will be run using the session)
        :param multitask_loss: tensorflow operator to get the result of the multitask loss. This info
        will be logged to be able to analyze it later
        """
        # Used to save and restore the model variables
        saver = tf.train.Saver()

        # If this model was already partially training before, load it from disk
        if self._config.get_model_load():
            # Restore variables from disk.
            saver.restore(self._sess, self._config.get_model_path())
            print("Model restored.")

        print("Starting training")
        training_start_time = time.time()

        iteration = 0
        learning_rate_manager = rm.LearningRateManager(
            self._config.get_learning_rate_initial_value(),
            self._config.get_learning_rate_manager_threshold(),
            self._config.get_learning_rate_manager_steps())

        for epoch in range(0, self._config.get_number_epochs()):
            print("Epoch: {0}".format(str(epoch)))
            # Training with all the PASCAL VOC records for each epoch
            # We train with 1 image per batch and 64 rois per image. From those 64, we'll use a max
            # of 16 foreground images. The rest will be background.
            training_reader = ds_reader.DatasetReader(
                training_batch_files,
                self._config.get_number_images_batch(),
                self._config.get_number_rois_per_image_batch(),
                self._config.get_number_max_foreground_rois_per_image_batch())
            training_batch = training_reader.get_batch()

            # Empty batch means we are done processing all images and rois for this epoch
            while training_batch != {}:
                _, loss = self._sess.run([training, multitask_loss], feed_dict={
                    self._image_input_batch: training_batch["images"],
                    self._roi_input_batch: training_batch["rois"],
                    self._class_label_batch: training_batch["class_labels"],
                    self._detection_label_batch: training_batch["reg_target_labels"],
                    self._learning_rate: learning_rate_manager.learning_rate
                })

                # Logging information about the multitask loss to be able to analyze it later
                output_analyzer.write_error_to_file(
                    self._config.get_training_error_file(), iteration, loss)
                # Adding error to learning rate manager so it can calculate when to reduce it
                learning_rate_manager.add_error(loss)

                iteration = iteration + 1

                training_batch = training_reader.get_batch()

            # Save model variables to disk
            if self._config.get_model_save():
                save_path = saver.save(self._sess, self._config.get_model_path())
                print("Model saved in path: {0} for epoch {1}".format(save_path, epoch))

        print("Done training. It took {0} minutes".format((time.time() - training_start_time) / 60))

    def test(self, prediction, test_batch_files):
        """
        This function detects and classifies objects in the given images

        :param prediction: tensorflow operator to detect objects (will be run using the session)
        """
        print("Starting prediction")
        prediction_start_time = time.time()

        # It generates batches from the list of test files
        test_reader = ds_reader.DatasetReader(
            test_batch_files,
            self._config.get_number_images_batch(),
            self._config.get_number_rois_per_image_batch(),
            self._config.get_number_max_foreground_rois_per_image_batch())
        test_batch = test_reader.get_batch()

        while test_batch != {}:
            predicted_classes = self._sess.run(prediction, feed_dict={
                self._image_input_batch: test_batch["images"],
                self._roi_input_batch: test_batch["rois"]
            })

            # Logging information about the prediction to be able to analyze it later
            output_analyzer.write_predictions_to_file(
                self._config.get_test_output_file(),
                test_batch["gt_objects"], np.transpose(predicted_classes, axes=[1, 0, 2]))
            test_batch = test_reader.get_batch()

        print("Done predicting. It took {0} minutes"
              .format((time.time() - prediction_start_time) / 60))


def run(properties_path, training_batch_files, test_batch_files):
    with tf.Session() as sess:
        config = conf_reader.ConfigReader(properties_path)
        rcnn_detection = RCNNDetection(sess, config)

        multitask_loss_op, training_op, prediction_op = rcnn_detection.get_net()

        # Initialization has to happen after defining the graph
        sess.run(tf.global_variables_initializer())

        # In order to be able to see the graph, we need to add this line after the graph is defined
        tf.summary.FileWriter(config.get_logs_path(), graph=tf.get_default_graph())

        rcnn_detection.train_net(training_op, multitask_loss_op, training_batch_files)
        rcnn_detection.test(prediction_op, test_batch_files)

        print("Run the command line:\n"
              "--> tensorboard --logdir=/tmp/tensorflow_logs "
              "\nThen open http://2usmtravesed.local:6006/ into your web browser")
