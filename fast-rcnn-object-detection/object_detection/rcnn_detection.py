import output_analyzer
import tensorflow as tf
import time
import learning_rate_manager as rm
import numpy as np
import dataset_reader as reader
import rcnn_net

EPOCHS = 15

NUMBER_CHANNELS = 3
CHANNEL_PIXELS = 600

NUMBER_RESNET_LAYERS = 15
NUMBER_HIDDEN_NODES = 800

# 20 different types of objects + background
NUMBER_CLASSES = 21
NUMBER_REGRESSION_FIELDS = 4

LOGS_PATH = '/tmp/tensorflow_logs/rcnn-detector/'

OUTPUT_FILE = "./output/result.txt"
ERROR_FILE = "./output/error.txt"

DIR_BATCH_FILES = "./dataset-rcnn/training/"

# TODO: Change the names once we generate the files
LIST_TRAINING_BATCH_FILES = \
    ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
LIST_TEST_BATCH_FILES = \
    ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

########
# Placeholders
########

# Shape of this placeholder is generally (2, 600, 600, 3), but when the last batch in a file is
# read, the batch size can be less than 2 (just one image left).
# That's why we specify None
# image_input_batch = tf.placeholder(
#     tf.float32, shape=(None, CHANNEL_PIXELS, CHANNEL_PIXELS, NUMBER_CHANNELS),
#     name="ImageInputBatch")

# We are only using one image per batch in this version
image_input_batch = tf.placeholder(
    tf.float32, shape=(CHANNEL_PIXELS, CHANNEL_PIXELS, NUMBER_CHANNELS),
    name="ImageInputBatch")

# Each RoI has 4 values (x, y, h, w)
roi_input_batch = tf.placeholder(
    tf.float32, shape=(None, NUMBER_REGRESSION_FIELDS), name="RoiInputBatch")

class_label_batch = tf.placeholder(
    tf.int32, shape=(None, NUMBER_CLASSES), name="ClassLabelsBatch")

detection_label_batch = tf.placeholder(
    tf.int32, shape=(None, NUMBER_REGRESSION_FIELDS), name="DetectionLabelsBatch")

learning_rate = tf.placeholder(tf.float32, name="LearningRate")


def train_net(session, training, multitask_loss):
    """
    This function trains the rcnn network

    :param session: tensorflow session to use to train the network
    :param training: tensorflow operator to train the network (will be run using the session)
    :param multitask_loss: tensorflow operator to get the result of the multitask loss. This info
    will be logged to be able to analyze it later
    """
    print "Starting training"
    training_start_time = time.time()

    iteration = 0
    learning_rate_manager = rm.LearningRateManager(0.001, 0.6, 80)

    for epoch in range(0, EPOCHS):
        print "Epoch: " + str(epoch)
        # Training with all the PASCAL VOC records for each epoch
        # We train with 1 image per batch and 64 rois per image. From those 64, we'll use a max of
        # 16 foreground images. The rest will be background.
        training_reader = reader.DatasetReader(LIST_TRAINING_BATCH_FILES, 1, 64, 16)
        training_batch = training_reader.get_batch()

        # Empty batch means we are done processing all images and rois for this epoch
        while training_batch != {}:
            _, loss = session.run([training, multitask_loss], feed_dict={
                image_input_batch: training_batch["images"],
                roi_input_batch: training_batch["rois"],
                class_label_batch: training_batch["class_labels"],
                detection_label_batch: training_batch["reg_target_labels"],
                learning_rate: learning_rate_manager.learning_rate
            })

            # Logging information about the multitask loss to be able to analyze it later
            output_analyzer.write_error_to_file(ERROR_FILE, iteration, loss)
            # Adding current error to learning rate manager so it can calculate when to reduce it
            learning_rate_manager.add_error(loss)

            iteration = iteration + 1

            training_batch = training_reader.get_batch()

    print "Done training. It took", (time.time() - training_start_time) / 60, "minutes"


def test(session, prediction):
    """
    This function detects and classifies objects in the given images

    :param session: tensorflow session to use to train the network
    :param prediction: tensorflow operator to detect objects (will be run using the session)
    """
    print "Starting prediction"
    prediction_start_time = time.time()

    # It generates batches from the list of test files
    test_reader = reader.DatasetReader(LIST_TEST_BATCH_FILES, 1, 64, 16)
    test_batch = test_reader.get_batch()

    while test_batch != {}:
        predicted_classes = session.run(prediction, feed_dict={
            image_input_batch: test_batch["images"],
            roi_input_batch: test_batch["rois"]
        })

        # Logging information about the prediction to be able to analyze it later
        output_analyzer.write_predictions_to_file(
            OUTPUT_FILE, test_batch["gt_objects"], np.transpose(predicted_classes, axes=[1, 0, 2]))
        test_batch = test_reader.get_batch()

    print "Done predicting. It took", (time.time() - prediction_start_time) / 60, "minutes"


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    multitask_loss_op, training_op, prediction_op = rcnn_net.get_net(
        NUMBER_CLASSES, NUMBER_REGRESSION_FIELDS, NUMBER_RESNET_LAYERS, NUMBER_HIDDEN_NODES,
        image_input_batch, roi_input_batch, class_label_batch, detection_label_batch, learning_rate)

    # In order to be able to see the graph, we need to add this line after the graph is defined
    tf.summary.FileWriter(LOGS_PATH, graph=tf.get_default_graph())

    train_net(sess, training_op, multitask_loss_op)
    test(sess, prediction_op)

    print("Run the command line:\n"
          "--> tensorboard --logdir=/tmp/tensorflow_logs "
          "\nThen open http://2usmtravesed.local:6006/ into your web browser")
