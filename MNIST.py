__author__ = 'HuPeng'

import numpy as np
import struct
class MNIST:

    def __init__(self, image_filename=None, label_filename=None):
        self.image_filename = image_filename
        self.label_filename = label_filename

    def readfile(self, filename):
        assert filename is not None
        bin_file = open(filename, 'rb')
        data = bin_file.read()
        bin_file.close()
        return data

    def image_read(self, image_filename=None):
        """
        read the image mat from the file
        :param image_filename:
        :return: 3D matrix [imageIdex [imageRow imageColums]]
        """
        if image_filename is None:
            data = self.readfile(self.image_filename)
        else:
            data = self.readfile(image_filename)
        magic, num_images, num_rows, num_columns = struct.unpack_from('>IIII', data)
        assert magic == 2051
        index = struct.calcsize('>IIII')
        type_nums = '>' + str(num_images * num_rows * num_columns) + 'B'
        image = struct.unpack_from(type_nums, data, index)
        # return np.array(image).reshape((numImage, numRow, numColumns))
        return (num_images, num_rows, num_columns), np.float64(np.array(image))

    def label_read(self, label_filename=None):
        """
        read the label vector from the file
        :param label_filename:
        :return: labels' row vector
        """
        if label_filename is None:
            data = self.readfile(self.label_filename)
        else:
            data = self.readfile(label_filename)
        magic, num_items = struct.unpack_from('>II', data)
        assert magic == 2049
        index = struct.calcsize('>II')
        type_nums = '>' + str(num_items) + 'B'
        labels = struct.unpack_from(type_nums, data, index)
        return num_items, np.array(labels)

    def filter(self, images, labels, filt):
        assert images.shape[0] == labels.shape[0]
        result_images = []
        result_labels = []
        for i in range(0, labels.shape[0]):
            if labels[i] in filt:
                result_images.append(images[i])
                result_labels.append(labels[i])
        return np.array(result_images), np.array(result_labels)

    def read_filter(self, filt, image_filename=None, label_filename=None):
        size_images, images = self.image_read(image_filename)
        size_images, labels = self.label_read(label_filename)
        if filt is not None:
            return self.filter(images, labels, filt)
        return images, labels

    def image_read_standardize(self, image_fimename=None):
        size, image_data = self.image_read(image_fimename)
        image_data = image_data.reshape((-1, size[1] * size[2])).T
        image_data /= image_data.max()
        std = np.std(image_data, axis=0)
        mean = np.mean(image_data, axis=0)
        return np.append(np.ones((1, size[0])), (image_data - mean) / (std + .1)).reshape((-1, size[0]))

    def read(self, image_filename=None, label_filename=None):
        return self.image_read(image_filename), self.label_read(label_filename)

    def read_standardize(self, image_filename=None, label_filename=None):
        image_data = self.image_read_standardize(image_filename)
        label_data = self.label_read(label_filename)[1]
        col = np.arange(image_data.shape[1])
        np.random.shuffle(col)
        return image_data[:, col], label_data[col]






