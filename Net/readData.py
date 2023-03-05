import os.path

import numpy as np
import torch
import file_reader as fr

import matplotlib.pyplot as plt

IMAGE_SIZE = 64  # 图像尺寸
LABEL_BYTES = 8 * 8  # 标签大小

SAMPLE_LENGTH = 64 * 64 + 8 * 8  # 每个样本尺寸


class DataSet(object):
    def __init__(self, images, labels, fake_data=False, dtype=torch.float32):
        if dtype not in (torch.uint8, torch.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                    'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]
            # images = images.astype(np.float32)
            # labels = labels.astype(np.int32)

        images = images.transpose(0, 3, 1, 2)
        images = torch.Tensor(images)
        labels = torch.Tensor(labels)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

    def next_batch_random(self, batch_size, fake_data=False):
        batch_size_valid = self._num_examples
        if batch_size <= self._num_examples:
            batch_size_valid = batch_size
        index_list = np.random.randint(0, self._num_examples, [batch_size_valid])
        return self._images[index_list], self._labels[index_list], self._qps[index_list]


def get_data_set(file_reader, read_bytes, is_loop=True, dtype=np.uint8, is_show_stat=False):
    data = file_reader.read_data(read_bytes, isloop=is_loop, dtype=dtype)
    data_bytes = len(data)
    assert data_bytes % SAMPLE_LENGTH == 0
    num_samples = int(data_bytes / SAMPLE_LENGTH)  # 样本数量
    data = data.reshape(num_samples, SAMPLE_LENGTH)
    images = data[:, 0:4096].astype(np.float32)
    images = np.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

    labels = np.zeros((num_samples, 64), dtype=np.int16)
    for i in range(num_samples):
        labels[i, :] = data[i, 4096:4096 + 64]

    newLabel = np.zeros((num_samples, 36, 64))
    for i in range(num_samples):
        for j in range(64):
            ind = labels[i, j]  # 第j个点的模式
            newLabel[i, ind, j] = 1

    labels = torch.Tensor(newLabel)
    if is_show_stat == True:
        print("")
        # range_stat = RangingStatistics(DEFAULT_THR_LIST, 'scalar')
        # count_list_ori, _ = range_stat.feed_data_list(labels)
        # print(count_list_ori)

    return DataSet(images, labels)


data_dir = 'Data/'  # path of training/validation/test data
trainPath = 'AI_Train_27300.dat'

if __name__ == '__main__':
    fileReader = fr.FileReader()
    fileReader.initialize(os.path.join(data_dir, trainPath), 13 * 7 * 4 * SAMPLE_LENGTH)
    Mdata = get_data_set(fileReader, 1 * (4096 + 64))
    labels = Mdata.labels
    print(labels.shape)
    # data_bytes = len(data)
    # # print(data_bytes)
    # assert data_bytes % SAMPLE_LENGTH == 0
    # num_samples = int(data_bytes / SAMPLE_LENGTH)  # 样本数量
    # images = data.reshape(13 * 7, 4096)
    # labels = data[:, 4096:4096 + 64].astype(np.float32)
    # print(images.shape)
    # n_line = 480 // 64
    # n_col = 832 // 64
    # img = np.zeros([448, 832])
    # for i_line in range(n_line):
    #     for i_col in range(n_col):
    #         data_img = images[i_line * 13 + i_col]
    #         img[i_line * 64: (i_line + 1) * 64, i_col * 64: (i_col + 1) * 64] = data_img.reshape(64, 64)
    #
    # # print(labels[273+13*2+11].reshape(8, 8))
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()
