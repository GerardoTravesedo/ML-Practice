import tensorflow as tf


class FileInputReader:

    # TODO: This only reads input in text CSV format: expand to images, text, ...
    # TODO: Add shuffling

    # This class needs to be instantiated and used within a TF session

    def __init__(self, session, file_queue, sample_queue, number_features, batch_size):
        """
        :param
            session: TF session to use when running the operation in this class
            file_queue: Instance of the queue to use for filename storage
            sample_queue: Instance of the queue to use for sample storage
            number_features: Number of features that each record contains
            batch_size: Size of the mini batches that it will retrieve from the queue
        """
        self._sess = session
        self._number_features = number_features
        self._batch_size = batch_size
        self._define_reader_graph(file_queue, sample_queue)

    def enqueue_file(self, filename):
        """
        This method enqueues a new file that will be used as the source of samples
        :param
            filename: The name of the file to enqueue
        """
        self._sess.run(self._enqueue_file, feed_dict={self._filename: filename})

    def get_sample_batch(self):
        """
        This method is used to dequeue a batch of samples from the input files
        :return:
            Tuple containing the next batch, the first element contains the batch of images
            and the second their respective labels
        """
        return self._sess.run(self._get_batch)

    def terminate_queue(self):
        """
        Terminates all the queue threads together using the coordinator
        """
        # Request the threads to stop
        self._coordinator.request_stop()
        # Waits until the thread stop
        self._coordinator.join(self._threads)

    def finish_enqueuing_files(self):
        """
        Closes the file queue so records can be retrieved from it. Call this method only when you
        are done queueing input files. The method reads all the lines in the files and enqueues
        them in the sample queue. These records will be used to build batched later
        """
        # Close the file queue so files can be dequeue from it and lines can be
        # sent to the sample queue
        self._sess.run(self._close_file_queue)
        # Add all the lines in the file to the sample queue
        try:
            while True:
                self._sess.run(self._read_file_line)
        except tf.errors.OutOfRangeError:
            print("All records have been enqueued")
            pass  # no more records
        # Close the sample queue since no more files are going to be queued and in preparation
        # to retrieve batches
        self._sess.run(self._close_sample_queue)

    def _define_reader_graph(self, file_queue, sample_queue):
        # Start populating the filename queue.
        self._coordinator = tf.train.Coordinator()
        # By default one enqueuer thread and one closer thread
        # The closer thread closes the queue when there is an exception
        self._threads = tf.train.start_queue_runners(coord=self._coordinator)

        # Placeholder to pass an input file to enqueue
        self._filename = tf.placeholder(tf.string, name="filename-input")

        # Setting up the filename queue
        self._file_queue = file_queue
        self._enqueue_file = self._file_queue.enqueue([self._filename], name="enqueue-file")
        self._close_file_queue = self._file_queue.close()

        # Setting up the sample queue
        self._sample_queue = sample_queue
        self._close_sample_queue = self._sample_queue.close()

        # Reading lines from files
        self._define_read_file_line_graph()

        # Getting a batch of records
        self._define_get_batch_graph()

    def _define_get_batch_graph(self):
        batch_size = self._batch_size
        # Dequeue batch_size records from the sample queue
        dequeue_batch = self._sample_queue.dequeue_many(batch_size, name="dequeue-batch")
        # Next batch features
        net_input_batch = tf.reshape(
            dequeue_batch[0], [batch_size, self._number_features], name="reshape-input")
        # Next batch label data
        net_target_batch = tf.reshape(
            dequeue_batch[1], [batch_size], name="reshape-target")
        self._get_batch = net_input_batch, net_target_batch

    def _define_read_file_line_graph(self):
        line_reader = tf.TextLineReader()
        # Return next record in the file
        key, value = line_reader.read(self._file_queue, name="read-line")
        # Default values for columns. We specify one default value per column
        # In our case there are 785 columns: label + 784 pixels
        record_default = [[0]] * (self._number_features + 1)
        # From string CSV to tensors (each column one tensor)
        fields = tf.decode_csv(value, record_defaults=record_default, name="decode-line")
        # Instead of having one tensor per pixel, we want one tensor with all the pixels
        pixels = tf.reshape(fields[1:], [self._number_features])
        target = fields[0]
        # Append record to sample queue
        self._read_file_line = self._sample_queue.enqueue([pixels, target], name="enqueue-line")
