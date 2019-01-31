import output_analyzer
import tensorflow as tf
import time
import learning_rate_manager as rm
import numpy as np
import reduced_resnet_builder
import rcnn_multitask_loss as mloss
import roi_pooling_layer
import dataset_reader as reader

EPOCHS = 15

NUMBER_CHANNELS = 3
CHANNEL_PIXELS = 600

NUMBER_RESNET_LAYERS = 15
NUMBER_HIDDEN_NODES = 1000

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
image_input_batch = tf.placeholder(
    tf.float32, shape=(None, CHANNEL_PIXELS, CHANNEL_PIXELS, NUMBER_CHANNELS),
    name="ImageInputBatch")
# Each RoI has 4 values (x, y, h, w)
roi_input_batch = tf.placeholder(
    tf.float32, shape=(None, NUMBER_REGRESSION_FIELDS), name="RoiInputBatch")

class_label_batch = tf.placeholder(
    tf.int32, shape=(None, NUMBER_CLASSES), name="ClassLabelsBatch")

detection_label_batch = tf.placeholder(
    tf.int32, shape=(None, NUMBER_REGRESSION_FIELDS), name="DetectionLabelsBatch")

learning_rate = tf.placeholder(tf.float32, name="LearningRate")

########
# Neural net
#######

he_init = tf.contrib.layers.variance_scaling_initializer()

resnet = reduced_resnet_builder.ReducedResnetBuilder(he_init) \
    .build_resnet(image_input_batch, NUMBER_RESNET_LAYERS)

roi_pooling_layer = roi_pooling_layer\
    .RoiPoolingLayer(resnet, roi_input_batch, 7, 7, 4).get_roi_pooling_layer()

pool2_flat = tf.reshape(roi_pooling_layer, [-1, 7 * 7 * 64])

fc_layer_1 = tf.layers.dense(
    pool2_flat, NUMBER_HIDDEN_NODES, activation=tf.nn.leaky_relu, kernel_initializer=he_init)

fc_layer_2 = tf.layers.dense(
    fc_layer_1, NUMBER_HIDDEN_NODES, activation=tf.nn.leaky_relu, kernel_initializer=he_init)

#######
# RoI classification branch
#######
class_fc = tf.layers.dense(
    fc_layer_2, NUMBER_CLASSES, activation=tf.nn.leaky_relu, kernel_initializer=he_init,
    name="Logits")
class_softmax = tf.nn.softmax(class_fc)

#######
# RoI detection branch
#######
detection_fc = tf.layers.dense(
    fc_layer_2, NUMBER_CLASSES, activation=tf.nn.leaky_relu, kernel_initializer=he_init)
# The output has to be 4 regression numbers for each class that is not background
detection_regressor = tf.layers.dense(
    detection_fc, NUMBER_REGRESSION_FIELDS * (NUMBER_CLASSES - 1),
    activation=tf.nn.leaky_relu, kernel_initializer=he_init, name="DetectionFields")
# So far we have all the regression targets together in a vector for all classes. We need to convert
# that into a matrix where rows represents classes and columns represent the predicted
# regression targets
detection_regressor_shape = tf.shape(detection_regressor)
detection_regressor_reshaped = tf.reshape(
    detection_regressor,
    [detection_regressor_shape[0], NUMBER_CLASSES - 1, NUMBER_REGRESSION_FIELDS])

#######
# Multi-task loss
#######

# Combined loss for classification and detection
multitask_loss = mloss.RCNNMultitaskLoss(
    class_predictions=class_softmax,
    detection_predictions=detection_regressor_reshaped,
    class_labels=class_label_batch,
    detection_labels=detection_label_batch)\
    .multitask_loss()

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(multitask_loss)

#######
# Testing
#######

# Axis = 1 because we want to find the max per row (horizontally)
max_class = tf.argmax(class_softmax, 1)

# In order to be able to see the graph, we need to add this line after the graph is defined
summary_writer = tf.summary.FileWriter(LOGS_PATH, graph=tf.get_default_graph())


def train_net(session):
    print "Starting training"
    training_start_time = time.time()

    iteration = 0
    learning_rate_manager = rm.LearningRateManager(0.001, 0.6, 80)

    for epoch in range(0, EPOCHS):
        print "Epoch: " + str(epoch)
        # Training with all the PASCAL VOC records for each epoch
        training_reader = reader.DatasetReader(LIST_TRAINING_BATCH_FILES)
        training_batch = training_reader.get_batch()

        # Empty batch means we are done processing all images and rois for this epoch
        while training_batch != {}:
            _, loss = session.run([training_op, multitask_loss], feed_dict={
                image_input_batch: training_batch["images"],
                roi_input_batch: training_batch["rois"],
                class_label_batch: training_batch["class_labels"],
                detection_label_batch: training_batch["reg_target_labels"],
                learning_rate: learning_rate_manager.learning_rate
            })

            output_analyzer.write_error_to_file(ERROR_FILE, iteration, loss)
            learning_rate_manager.add_error(loss)

            iteration = iteration + 1

            training_batch = training_reader.get_batch()

    print "Done training. It took", (time.time() - training_start_time) / 60, "minutes"


def test(session):
    print "Starting prediction"
    prediction_start_time = time.time()

    test_reader = reader.DatasetReader(LIST_TEST_BATCH_FILES)
    test_batch = test_reader.get_batch()

    while test_batch != {}:
        # TODO: We need non max supression
        predicted_classes = session.run(max_class, feed_dict={
            image_input_batch: test_batch["images"],
            roi_input_batch: test_batch["rois"],
            class_label_batch: test_batch["class_labels"],
            detection_label_batch: test_batch["reg_target_labels"]
        })

        # TODO: review this class
        output_analyzer.write_predictions_to_file(
            OUTPUT_FILE, test_batch["labels"], np.transpose(predicted_classes))
        test_batch = test_reader.get_batch()

    print "Done predicting. It took", (time.time() - prediction_start_time) / 60, "minutes"


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_net(sess)
    test(sess)

    print("Run the command line:\n"
          "--> tensorboard --logdir=/tmp/tensorflow_logs "
          "\nThen open http://2usmtravesed.local:6006/ into your web browser")
