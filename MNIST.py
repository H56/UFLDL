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
        magic, numImage, numRow, numColumns = struct.unpack_from('>IIII', data)
        assert magic == 2051
        index = struct.calcsize('>IIII')
        type_nums = '>' + str(numImage * numRow * numColumns) + 'B'
        image = struct.unpack_from(type_nums, data, index)
        return np.array(image).reshape((numImage, numRow, numColumns))

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
        magic, numItems = struct.unpack_from('>II', data)
        assert magic == 2049
        index = struct.calcsize('>II')
        type_nums = '>' + str(numItems) + 'B'
        labels = struct.unpack_from(type_nums, data, index)
        return np.array(labels)

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
        images = self.image_read(image_filename)
        labels = self.label_read(label_filename)
        if filt is not None:
            return self.filter(images, labels, filt)
        return images, labels






