import InputFileReader
import tensorflow as tf

logs_path = '/tmp/tensorflow_logs/input-reader/'

with tf.Session() as sess:
    filename_queue = tf.FIFOQueue(capacity=5, dtypes=[tf.string], name="filename_queue")

    instance_queue = tf.FIFOQueue(
      capacity=10,
      dtypes=[tf.int32, tf.int32],
      shapes=[[3], []],
      name="sample_queue")

    reader = InputFileReader.FileInputReader(
      session=sess, file_queue=filename_queue, sample_queue=instance_queue,
      number_features=3, batch_size=2)

    reader.enqueue_file("test/input-file-reader-test.txt")
    reader.finish_enqueuing_files()

    batch = reader.get_sample_batch()
    print(batch)

    reader.terminate_queue()

    # In order to be able to see the graph, we need to add this line after the graph is defined
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    print("Run the command line:\n"
          "--> tensorboard --logdir=/tmp/tensorflow_logs "
          "\nThen open http://2usmtravesed.local:6006/ into your web browser")



