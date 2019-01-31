import DatasetReader
import OutputAnalyzer
import tensorflow as tf
import time
import LearningRateManager
import numpy as np
import ResnetCifar10

BATCH_SIZE = 64
EPOCHS = 15

NUMBER_CHANNELS = 3
CHANNEL_PIXELS = 32

NUMBER_RESNET_LAYERS = 20

LAST_FC_UNITS = 1000
NUMBER_CLASSES = 10

LOGS_PATH = '/tmp/tensorflow_logs/cifar-10-classifier/'

OUTPUT_FILE = "./output/result.txt"
ERROR_FILE = "./output/error.txt"

DIR_BATCH_FILES = "./dataset-reshaped-normalized/"

LIST_BATCH_FILES = \
  ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

# Shape of this placeholder is generally (256, 224, 224, 3), but when the last batch in a file is
# read, the batch size can be different than 256 (only a few records left in the file).
# That's why we specify None
input_batch = tf.placeholder(
  tf.float32, shape=(None, CHANNEL_PIXELS, CHANNEL_PIXELS, NUMBER_CHANNELS), name="InputBatch")
label_batch = tf.placeholder(tf.int32, shape=None, name="LabelsBatch")

learning_rate = tf.placeholder(tf.float32, name="LearningRate")

he_init = tf.contrib.layers.variance_scaling_initializer()

resnet = ResnetCifar10.ResnetCifar10(he_init).build_resnet(input_batch, NUMBER_RESNET_LAYERS)

# Global average pool
last_avg_pool = tf.nn.avg_pool(
  resnet, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding="VALID", name="FinalAvgPool")

pool2_flat = tf.reshape(last_avg_pool, [-1, 1 * 1 * 64])

logits = tf.layers.dense(
  pool2_flat,
  NUMBER_CLASSES,
  activation=tf.nn.leaky_relu,
  kernel_initializer=he_init,
  name="Logits")

# This indicates if the target is the top prediction (highest prob)
# for the input
correct = tf.nn.in_top_k(logits, label_batch, 1)

# Softmax predicts only one class at a time
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  labels=label_batch, logits=logits)

error = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(error)

softmax = tf.nn.softmax(logits)
# Axis = 1 because we want to find the max per row (horizontally)
max_class = tf.argmax(softmax, 1)

# In order to be able to see the graph, we need to add this line after the graph is defined
summary_writer = tf.summary.FileWriter(LOGS_PATH, graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print "Starting training"
    training_start_time = time.time()

    iteration = 0
    learning_rate_manager = LearningRateManager.LearningRate(0.001, 0.6, 80)

    for epoch in range(0, EPOCHS):
        print "Epoch: " + str(epoch)
        # Training with all the cifar10 files for each epoch
        for cifar10_file in LIST_BATCH_FILES:
            print "File " + cifar10_file

            reader = DatasetReader.DatasetReader(DIR_BATCH_FILES + cifar10_file, BATCH_SIZE)
            batch = reader.get_batch()

            while batch != {}:
                _, loss = sess.run([training_op, error], feed_dict={
                  input_batch: batch["data"],
                  label_batch: batch["labels"],
                  learning_rate: learning_rate_manager.learning_rate
                })

                OutputAnalyzer.write_error_to_file(ERROR_FILE, iteration, loss)
                learning_rate_manager.add_error(loss)

                iteration = iteration + 1

                batch = reader.get_batch()

    print "Done training. It took", (time.time() - training_start_time) / 60, "minutes"

    print "Starting prediction"
    prediction_start_time = time.time()

    reader = DatasetReader.DatasetReader(DIR_BATCH_FILES + "test_batch", BATCH_SIZE)
    batch = reader.get_batch()

    while batch != {}:
        predicted_classes = sess.run(max_class, feed_dict={
          input_batch: batch["data"],
          label_batch: batch["labels"]
        })

        OutputAnalyzer.write_predictions_to_file(
          OUTPUT_FILE, batch["labels"], np.transpose(predicted_classes))
        batch = reader.get_batch()

    print "Done predicting. It took", (time.time() - prediction_start_time) / 60, "minutes"

    print("Run the command line:\n"
          "--> tensorboard --logdir=/tmp/tensorflow_logs "
          "\nThen open http://2usmtravesed.local:6006/ into your web browser")

