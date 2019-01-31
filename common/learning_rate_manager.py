import numpy as np


class LearningRateManager:

    def __init__(self, initial_rate, threshold, number_steps):
        """
        Initializes a LearningRate object

        :param
            initial_rate: initial learning rate to use at the beginning of training

            threshold: when the standard deviation calculated for the last few elements is
            less or equal than this value, it reduces the learning rate to converge better

            number_steps: number of consecutive step errors to use in the calculation of the
            standard deviation to determine if the errors values are too close and there is a
            plateau
        """
        self.learning_rate = initial_rate
        self.threshold = threshold
        self.number_steps = number_steps
        self.previous_errors = []

    def add_error(self, error):
        # Concatenating new error to list of errors
        self.previous_errors = self.previous_errors + [error]

        # If there are more than N elements, clip the list to keep just the last N, where N is the
        # number of errors to use for learning rate calculation
        if len(self.previous_errors) > self.number_steps:
            self.previous_errors = self.previous_errors[-self.number_steps:]
            # Calculating standard deviation from last N errors added to the list
            std = np.std(self.previous_errors)
            print str(std)

            # If standard deviation of errors is very little, error has plateau and we
            # decrease learning rate
            # Also we don't want to decrease the learning rate too much so it doesn't converge very
            # slowly
            if std <= self.threshold and self.learning_rate > 0.00001:
                new_learning_rate = self.learning_rate / 10
                print "Reducing learning rate from " + \
                      str(self.learning_rate) + " to " + str(new_learning_rate)
                self.learning_rate = new_learning_rate
                # Once we reduce the learning rate we clean the previous errors, to start from
                # scratch with the new rate
                self.previous_errors = []

        print str(self.previous_errors)

