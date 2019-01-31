import tensorflow as tf


class ResnetCifar10:

    def __init__(self, kernel_initializer):
        self.kernel_initializer = kernel_initializer

    def build_resnet(self, input_batch, number_layers):
        net_info = self._get_net_info(number_layers)

        init_conv = tf.layers.conv2d(
          input_batch, filters=16, kernel_size=3, strides=[1, 1],
          padding="SAME", activation=None, name="InitialConvLayer",
          kernel_initializer=self.kernel_initializer)
        init_batch_norm = tf.layers.batch_normalization(init_conv, name="InitialConvBatch")
        init_activation = tf.nn.leaky_relu(init_batch_norm, name="InitialConvRelu")

        input_next_unit = init_activation

        for residual_unit_set_info in net_info:
            number_filters = residual_unit_set_info[0]
            number_units = residual_unit_set_info[1]

            for unit in (1, number_units):
                if unit == 1 and number_filters != 16:
                    input_next_unit = self._add_residual_unit_with_changing_size(
                      input_next_unit, number_filters, unit)
                else:
                    input_next_unit = self._add_residual_unit(input_next_unit, number_filters, unit)

        return input_next_unit

    # TODO: Understand why batch normalization before activation
    def _add_residual_unit(self, input_batch, number_filters, unit_number):
        identifier = "Filters" + str(number_filters) + "-Unit" + str(unit_number)

        conv_1 = tf.layers.conv2d(
          input_batch, filters=number_filters, kernel_size=3, strides=[1, 1],
          padding="SAME", activation=None, name=identifier + "-Conv1",
          kernel_initializer=self.kernel_initializer)
        batch_norm_1 = tf.layers.batch_normalization(conv_1, name=identifier + "-Batch1")
        activation_1 = tf.nn.leaky_relu(batch_norm_1, name=identifier + "-Relu1")

        conv_2 = tf.layers.conv2d(
          activation_1, filters=number_filters, kernel_size=3, strides=[1, 1],
          padding="SAME", activation=None, name=identifier + "-Conv2",
          kernel_initializer=self.kernel_initializer)
        batch_norm_2 = tf.layers.batch_normalization(conv_2, name=identifier + "-Batch2")

        addition = tf.add(input_batch, batch_norm_2, name=identifier + "-Add")

        return tf.nn.leaky_relu(addition, name=identifier + "-Relu2")

    def _add_residual_unit_with_changing_size(self, input_batch, number_filters, unit_number):
        identifier = "Filters" + str(number_filters) + "-Unit" + str(unit_number)

        conv_1 = tf.layers.conv2d(
          input_batch, filters=number_filters, kernel_size=3, strides=[2, 2],
          padding="SAME", activation=None, name=identifier + "-Conv1",
          kernel_initializer=self.kernel_initializer)
        batch_norm_1 = tf.layers.batch_normalization(conv_1, name=identifier + "-Batch1")
        activation_1 = tf.nn.leaky_relu(batch_norm_1, name=identifier + "-Relu1")

        conv_size_change = tf.layers.conv2d(
          input_batch, filters=number_filters, kernel_size=1, strides=[2, 2],
          padding="SAME", activation=None, name=identifier + "-ConvAdjustSize",
          kernel_initializer=self.kernel_initializer)
        batch_norm_size_change = \
            tf.layers.batch_normalization(conv_size_change, name=identifier + "-BatchAdjustSize")

        conv_2 = tf.layers.conv2d(
          activation_1, filters=number_filters, kernel_size=3, strides=[1, 1],
          padding="SAME", activation=None, name=identifier + "-Conv2",
          kernel_initializer=self.kernel_initializer)
        batch_norm_2 = tf.layers.batch_normalization(conv_2, name=identifier + "-Batch2")

        addition = tf.add(batch_norm_size_change, batch_norm_2, name=identifier + "-Add")

        return tf.nn.leaky_relu(addition, name=identifier + "-Relu2")

    def _get_net_info(self, number_layers):
        if number_layers == 20:
            return self._get_20_layers_net_info()
        else:
            raise Exception("Invalid number of layers for resnet")

    # Use for ImageNet
    def _get_20_layers_net_info(self):
        return [(16, 5), (32, 5), (64, 5)]
