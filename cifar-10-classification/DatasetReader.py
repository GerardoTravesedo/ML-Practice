import pickle
import numpy as np


class DatasetReader:

    def __init__(self,  filename, batch_size):
        with open(filename, 'rb') as fo:
            # The dictionary returned by load method contains the following fields:
            # data --
            #    A 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32
            #    colour image.
            #    The first 1024 entries contain the red channel values, the next 1024 the green,
            #    and the final 1024 the blue.
            #    The image is stored in row-major order, so that the first 32 entries of the array
            #    are the red channel values of the first row of the image.
            # labels --
            #    A list of 10000 numbers in the range 0-9. The number at index i indicates the label
            #    of the ith image in the array data.
            self.data = pickle.load(fo)
        # next_records represents the row that marks the beginning of the next batch
        self.next_record = 0
        self.batch_size = batch_size
        self.total_records = np.size(self.data["data"], 0)

    def get_batch(self):
        # Getting range of rows with size batch_size
        if self.next_record >= self.total_records:
            return {}

        if self.total_records - self.next_record < self.batch_size:
            records_to_fetch = self.total_records - self.next_record
        else:
            records_to_fetch = self.batch_size

        data_batch = self.data["data"][self.next_record:self.next_record + records_to_fetch]
        label_batch = self.data["labels"][self.next_record:self.next_record + records_to_fetch]
        print "Batch: {size:", self.batch_size, ", initial_index:", self.next_record, \
            ", final_index:", self.next_record + records_to_fetch, "}"

        self.next_record = self.next_record + self.batch_size
        return {"data": data_batch, "labels": label_batch}
