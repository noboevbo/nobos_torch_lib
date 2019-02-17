import random

import numpy as np
from nobos_commons.data_structures.constants.dataset_split import DatasetSplit
from nobos_commons.data_structures.dimension import ImageSize
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class EhpiDataset(Dataset):
    def __init__(self, dataset_path: str, dataset_split: DatasetSplit = DatasetSplit.TRAIN,
                 transform: Compose = None):
        if dataset_split == DatasetSplit.TRAIN:
            x_path = dataset_path + "X_train.txt"
            y_path = dataset_path + "Y_train.txt"
        else:
            x_path = dataset_path + "X_test.txt"
            y_path = dataset_path + "Y_test.txt"
        self.x = self.load_X(x_path)
        self.y = self.load_y(y_path)
        self.transform = transform

        self.__length = len(self.y)

    def load_X(self, X_path):
        file = open(X_path, 'r')
        rows = []
        for row in file:
            array = np.asarray(list(map(float, row.split(','))))
            array = np.reshape(array, (18, 3))
            # array = np.pad(array, [(7, 7), (0, 0)], mode='constant')
            rows.append(array)
        X_ = np.array(rows, dtype=np.float32)
        blocks = int(len(X_) / 32)
        X_ = np.array(np.split(X_, blocks))

        # Remove joint positions outside of the image
        X_[X_ < 0] = 0
        X_[X_ > 1] = 1
        # Transpose to SequenceNum, Channel, Width, Height
        X_ = np.transpose(X_, (0, 3, 1, 2))
        # Set last channel (currently containing the joint probability) to zero
        X_[:, 2, :, :] = 0

        return X_

        # Load the networks outputs

    def load_y(self, y_path):
        file = open(y_path, 'r')
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
            dtype=np.int32
        )
        file.close()

        # for 0-based indexing
        if y_.min() > 0:
            y_ = y_ - 1

        return y_

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        sample = {"x": self.x[index], "y": self.y[index]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ScaleEhpi(object):
    def __init__(self, image_size: ImageSize):
        if image_size.width > image_size.height:
            self.image_size = ImageSize(width=1, height=image_size.height/image_size.width)
        else:
            self.image_size = ImageSize(width=image_size.width / image_size.height, height=1)
        # self.image_size = image_size

    def __call__(self, sample):
        ehpi_img = sample['x']
        curr_min_x = np.min(ehpi_img[0, :, :])
        curr_min_y = np.min(ehpi_img[1, :, :])
        curr_max_x = np.max(ehpi_img[0, :, :])
        curr_max_y = np.max(ehpi_img[1, :, :])
        max_factor_x = self.image_size.width / curr_max_x
        max_factor_y = self.image_size.height / curr_max_y
        min_factor_x = (self.image_size.width * 0.1) / (curr_max_x - curr_min_x)
        min_factor_y = (self.image_size.height * 0.1) / (curr_max_y - curr_min_y)
        min_factor = max(min_factor_x, min_factor_y)
        max_factor = min(max_factor_x, max_factor_y)
        factor = random.uniform(min_factor, max_factor)
        ehpi_img[0, :, :] = ehpi_img[0, :, :] * factor
        ehpi_img[1, :, :] = ehpi_img[1, :, :] * factor
        sample['x'] = ehpi_img
        return sample


class TranslateEhpi(object):
    def __init__(self, image_size: ImageSize):
        if image_size.width > image_size.height:
            self.image_size = ImageSize(width=1, height=image_size.height/image_size.width)
        else:
            self.image_size = ImageSize(width=image_size.width / image_size.height, height=1)

    def __call__(self, sample):
        ehpi_img = sample['x']
        max_minus_translate_x = -np.min(ehpi_img[0, :, :])
        max_minus_translate_y = -np.min(ehpi_img[1, :, :])
        max_plus_translate_x = self.image_size.width - np.max(ehpi_img[0, :, :])
        max_plus_translate_y = self.image_size.height - np.max(ehpi_img[1, :, :])
        translate_x = random.uniform(max_minus_translate_x, max_plus_translate_x)
        translate_y = random.uniform(max_minus_translate_y, max_plus_translate_y)
        ehpi_img[0, :, :] = ehpi_img[0, :, :] + translate_x
        ehpi_img[1, :, :] = ehpi_img[1, :, :] + translate_y
        sample['x'] = ehpi_img
        return sample


class NormalizeEhpi(object):
    def __init__(self, image_size: ImageSize):
        if image_size.width > image_size.height:
            self.image_size = ImageSize(width=1, height=image_size.height/image_size.width)
        else:
            self.image_size = ImageSize(width=image_size.width / image_size.height, height=1)

    def __call__(self, sample):
        ehpi_img = sample['x']
        curr_min_x = np.min(ehpi_img[0, :, :])
        curr_min_y = np.min(ehpi_img[1, :, :])
        # curr_max_x = np.max(ehpi_img[0, :, :])
        # curr_max_y = np.max(ehpi_img[1, :, :])
        ehpi_img[0, :, :] = ehpi_img[0, :, :] - curr_min_x
        ehpi_img[1, :, :] = ehpi_img[1, :, :] - curr_min_y
        max_factor_x = self.image_size.width / np.max(ehpi_img[0, :, :])
        max_factor_y = self.image_size.width / np.max(ehpi_img[1, :, :])
        ehpi_img[0, :, :] = ehpi_img[0, :, :] * max_factor_x
        ehpi_img[1, :, :] = ehpi_img[1, :, :] * max_factor_y
        sample['x'] = ehpi_img
        return sample


class FlipEhpi(object):
    def __init__(self, with_scores: bool = True):
        self.step_size = 3 if with_scores else 2
        self.left_indexes = [5, 6, 7, 11, 12, 13, 15, 17]
        self.right_indexes = [2, 3, 4, 8, 9, 10, 14, 16]

    def __call__(self, sample):
        if bool(random.getrandbits(1)):
            return sample
        ehpi_img = sample['x']
        curr_min_x = np.min(ehpi_img[0, :, :])
        # curr_min_y = np.min(ehpi_img[1, :, :])
        curr_max_x = np.max(ehpi_img[0, :, :])
        # curr_max_y = np.max(ehpi_img[1, :, :])
        # reflect_y = (curr_max_y + curr_min_y) / 2
        reflect_x = (curr_max_x + curr_min_x) / 2
        # ehpi_img[1, :, :] = reflect_y - (ehpi_img[1, :, :] - reflect_y)
        ehpi_img[0, :, :] = reflect_x - (ehpi_img[0, :, :] - reflect_x)

        if bool(random.getrandbits(1)):
            # Swap Left / Right joints
            for left_index, right_index in zip(self.left_indexes, self.right_indexes):
                tmp = np.copy(ehpi_img)
                ehpi_img[0:, :, left_index] = tmp[0:, :, right_index]
                ehpi_img[1:, :, left_index] = tmp[1:, :, right_index]
                ehpi_img[2:, :, left_index] = tmp[2:, :, right_index]
                ehpi_img[0:, :, right_index] = tmp[0:, :, left_index]
                ehpi_img[1:, :, right_index] = tmp[1:, :, left_index]
                ehpi_img[2:, :, right_index] = tmp[2:, :, left_index]
        sample['x'] = ehpi_img
        return sample
