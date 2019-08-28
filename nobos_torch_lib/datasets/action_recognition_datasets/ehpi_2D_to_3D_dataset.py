import os
import random
from typing import List, Dict

import numpy as np
import torch
from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.tools.log_handler import logger
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose


DEPTH = 10

class Ehpi2Dto3DDataset(Dataset):
    def __init__(self, dataset_path: str, dataset_part: DatasetPart = DatasetPart.TRAIN,
                 transform: Compose = None, num_joints: int = 18):
        self.dataset_path = dataset_path
        self.X_path = os.path.join(dataset_path, "X_{}.csv".format(dataset_part.name.lower()))
        self.y_path = os.path.join(dataset_path, "Y_{}.csv".format(dataset_part.name.lower()))
        self.num_joints = num_joints
        self.x = self.load_X()
        self.y_sequence_ids, self.y_labels = self.load_y()
        self.transform = transform
        assert(len(self.x) == len(self.y_labels) == len(self.y_sequence_ids)), "Unequal Dataset size and labels. Data: {}, Labels: {}, IDS: {}, Path: {}".format(
            len(self.x), len(self.y_labels), len(self.y_sequence_ids), self.dataset_path
        )
        self.__length = len(self.y_labels)
        self.print_label_statistics()

    # in old: y: [action_id, actuin_UID]
    def print_label_statistics(self):
        for label, count in self.get_label_statistics().items():
            print("Sequence: '{}' - Images: '{}'".format(label, count))

    def get_label_statistics(self) -> Dict[int, int]:
        unique, counts = np.unique(self.y_sequence_ids, return_counts=True)
        return dict(zip(unique, counts))

    def load_X(self):
        X_ = np.loadtxt(self.X_path, delimiter=',', dtype=np.float32)
        X_ = np.reshape(X_, (X_.shape[0], 32, self.num_joints, 3))

        X_ = np.transpose(X_, (0, 3, 1, 2))

        return X_

    def load_y(self):
        y_temp = np.loadtxt(self.y_path, delimiter=',', dtype=np.float32)
        sequence_ids = y_temp[:, 0].astype('int32')
        labels = y_temp[:, 1:]
        labels = np.reshape(labels, (labels.shape[0], 32, self.num_joints, 3))
        labels = np.transpose(labels, (0, 3, 1, 2))
        return sequence_ids, labels

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        sample = {"x": self.x[index].copy(), "y": self.y_labels[index], "seq": self.y_sequence_ids[index]}
        if self.transform:
            try:
                sample = self.transform(sample)
            except Exception as err:
                logger.error("Error transform. Dataset: {}, index: {}, x_min_max: {}/{}".format(
                    self.dataset_path, index, self.x[index].min(), self.x[index].max()))
                logger.error(err)
                raise err
        return sample


class RemoveJointsOutsideImgEhpi2Dto3D(object):
    def __init__(self, image_size: ImageSize):
        # if image_size.width > image_size.height:
        #     self.image_size = ImageSize(width=1, height=image_size.height/image_size.width)
        # else:
        #     self.image_size = ImageSize(width=image_size.width / image_size.height, height=1)
        self.image_size = image_size

    def __call__(self, sample):
        sample['x'] = self.__removeJointsOutsideImg(sample['x'])
        sample['y'] = self.__removeJointsOutsideImg(sample['y'])
        return sample

    def __removeJointsOutsideImg(self, ehpi_img):
        tmp = np.copy(ehpi_img)
        for i in range(0, 3):
            ehpi_img[i, :, :][tmp[0, :, :] > self.image_size.width] = 0
            ehpi_img[i, :, :][tmp[0, :, :] < 0] = 0
            ehpi_img[i, :, :][tmp[1, :, :] > self.image_size.height] = 0
            ehpi_img[i, :, :][tmp[1, :, :] < 0] = 0
        return ehpi_img

class ScaleEhpi3D(object):
    def __init__(self, image_size: ImageSize):
        # if image_size.width > image_size.height:
        #     self.image_size = ImageSize(width=1, height=image_size.height/image_size.width)
        # else:
        #     self.image_size = ImageSize(width=image_size.width / image_size.height, height=1)
        self.image_size = image_size

    def __call__(self, sample):
        ehpi_img = sample['x']
        tmp = np.copy(ehpi_img)
        curr_min_x = np.min(ehpi_img[0, :, :][ehpi_img[0, :, :] > 0])
        curr_min_y = np.min(ehpi_img[1, :, :][ehpi_img[1, :, :] > 0])
        curr_min_z = np.min(ehpi_img[2, :, :][ehpi_img[2, :, :] > 0])
        curr_max_x = np.max(ehpi_img[0, :, :])
        curr_max_y = np.max(ehpi_img[1, :, :])
        curr_max_z = np.max(ehpi_img[2, :, :])
        max_factor_x = self.image_size.width / curr_max_x
        max_factor_y = self.image_size.height / curr_max_y
        max_factor_z = DEPTH / curr_max_z
        min_factor_x = (self.image_size.width * 0.1) / (curr_max_x - curr_min_x)
        min_factor_y = (self.image_size.height * 0.1) / (curr_max_y - curr_min_y)
        min_factor_z = (DEPTH * 0.1) / (curr_max_z - curr_min_z)
        min_factor = max(min_factor_x, min_factor_y, min_factor_z)
        max_factor = min(max_factor_x, max_factor_y, max_factor_z)
        factor = random.uniform(min_factor, max_factor)
        for i in range(0, 3):
            ehpi_img[i, :, :] = ehpi_img[i, :, :] * factor
            ehpi_img[i, :, :][tmp[i, :, :] == 0] = 0
        sample['x'] = ehpi_img
        return sample


class TranslateEhpi3D(object):
    def __init__(self, image_size: ImageSize):
        self.image_size = image_size
        # if image_size.width > image_size.height:
        #     self.image_size = ImageSize(width=1, height=image_size.height/image_size.width)
        # else:
        #     self.image_size = ImageSize(width=image_size.width / image_size.height, height=1)

    def __call__(self, sample):
        ehpi_img = sample['x']
        tmp = np.copy(ehpi_img)
        max_minus_translate_x = -np.min(ehpi_img[0, :, :][ehpi_img[0, :, :] > 0])
        max_minus_translate_y = -np.min(ehpi_img[1, :, :][ehpi_img[1, :, :] > 0])
        max_minus_translate_z = -np.min(ehpi_img[2, :, :][ehpi_img[2, :, :] > 0])
        max_plus_translate_x = self.image_size.width - np.max(ehpi_img[0, :, :])
        max_plus_translate_y = self.image_size.height - np.max(ehpi_img[1, :, :])
        max_plus_translate_z = self.image_size.height - np.max(ehpi_img[2, :, :])
        translate_x = random.uniform(max_minus_translate_x, max_plus_translate_x)
        translate_y = random.uniform(max_minus_translate_y, max_plus_translate_y)
        translate_z = random.uniform(max_minus_translate_z, max_plus_translate_z)
        # TODO: Translate only joints which are not 0..
        ehpi_img[0, :, :] = ehpi_img[0, :, :] + translate_x
        ehpi_img[1, :, :] = ehpi_img[1, :, :] + translate_y
        ehpi_img[2, :, :] = ehpi_img[2, :, :] + translate_z
        ehpi_img[0, :, :][tmp[0, :, :] == 0] = 0
        ehpi_img[1, :, :][tmp[1, :, :] == 0] = 0
        ehpi_img[2, :, :][tmp[2, :, :] == 0] = 0
        sample['x'] = ehpi_img
        return sample


class NormalizeEhpi3D(object):
    def __init__(self, image_size: ImageSize):
        if image_size.width > image_size.height:
            self.aspect_ratio = ImageSize(width=1, height=image_size.height / image_size.width)
        else:
            self.aspect_ratio = ImageSize(width=image_size.width / image_size.height, height=1)

    def __call__(self, sample):
        sample['x'] = self.__normalize2D(sample['x'])
        sample['y'] = self.__normalize3D(sample['y'])
        return sample

    def __normalize3D(self, ehpi_img):
        tmp = np.copy(ehpi_img)
        curr_min_x = np.min(ehpi_img[0, :, :][ehpi_img[0, :, :] > 0])
        curr_min_y = np.min(ehpi_img[1, :, :][ehpi_img[1, :, :] > 0])
        curr_min_z = np.min(ehpi_img[2, :, :][ehpi_img[2, :, :] > 0])

        # Set x start to 0
        ehpi_img[0, :, :] = ehpi_img[0, :, :] - curr_min_x
        # Set y start to 0
        ehpi_img[1, :, :] = ehpi_img[1, :, :] - curr_min_y
        # Set z start to 0
        ehpi_img[2, :, :] = ehpi_img[2, :, :] - curr_min_z

        # Set x to max image_size.width
        max_factor_x = 1 / np.max(ehpi_img[0, :, :])
        max_factor_y = 1 / np.max(ehpi_img[1, :, :])
        max_factor_z = 1 / np.max(ehpi_img[2, :, :])
        ehpi_img[0, :, :] = ehpi_img[0, :, :] * max_factor_x
        ehpi_img[1, :, :] = ehpi_img[1, :, :] * max_factor_y
        ehpi_img[2, :, :] = ehpi_img[2, :, :] * max_factor_z
        ehpi_img[0, :, :][tmp[0, :, :] == 0] = 0
        ehpi_img[1, :, :][tmp[1, :, :] == 0] = 0
        ehpi_img[2, :, :][tmp[2, :, :] == 0] = 0
        # test = ehpi_img[0, :, :].max()
        # test2 = ehpi_img[1, :, :].max()
        return ehpi_img

    def __normalize2D(self, ehpi_img):
        tmp = np.copy(ehpi_img)
        curr_min_x = np.min(ehpi_img[0, :, :][ehpi_img[0, :, :] > 0])
        curr_min_y = np.min(ehpi_img[1, :, :][ehpi_img[1, :, :] > 0])

        # Set x start to 0
        ehpi_img[0, :, :] = ehpi_img[0, :, :] - curr_min_x
        # Set y start to 0
        ehpi_img[1, :, :] = ehpi_img[1, :, :] - curr_min_y

        # Set x to max image_size.width
        max_factor_x = 1 / np.max(ehpi_img[0, :, :])
        max_factor_y = 1 / np.max(ehpi_img[1, :, :])
        ehpi_img[0, :, :] = ehpi_img[0, :, :] * max_factor_x
        ehpi_img[1, :, :] = ehpi_img[1, :, :] * max_factor_y
        ehpi_img[0, :, :][tmp[0, :, :] == 0] = 0
        ehpi_img[1, :, :][tmp[1, :, :] == 0] = 0
        # test = ehpi_img[0, :, :].max()
        # test2 = ehpi_img[1, :, :].max()
        return ehpi_img


# class FlipEhpi(object):
#     def __init__(self, with_scores: bool = True, left_indexes: List[int] = [4, 6, 7, 8, 12, 13, 14],
#                  right_indexes: List[int] = [5, 9, 10, 11, 15, 16, 17]):
#         self.step_size = 3 if with_scores else 2
#         self.left_indexes = left_indexes
#         self.right_indexes = right_indexes
#
#     def __call__(self, sample):
#         if bool(random.getrandbits(1)):
#             return sample
#         ehpi_img = sample['x']
#         tmp = np.copy(ehpi_img)
#         curr_min_x = np.min(ehpi_img[0, :, :][ehpi_img[0, :, :] > 0])
#         # curr_min_y = np.min(ehpi_img[1, :, :])
#         curr_max_x = np.max(ehpi_img[0, :, :])
#         # curr_max_y = np.max(ehpi_img[1, :, :])
#         # reflect_y = (curr_max_y + curr_min_y) / 2
#         reflect_x = (curr_max_x + curr_min_x) / 2
#         # ehpi_img[1, :, :] = reflect_y - (ehpi_img[1, :, :] - reflect_y)
#         ehpi_img[0, :, :] = reflect_x - (ehpi_img[0, :, :] - reflect_x)
#         ehpi_img[0, :, :][tmp[0, :, :] == 0] = 0
#         ehpi_img[1, :, :][tmp[1, :, :] == 0] = 0
#         if bool(random.getrandbits(1)):
#             # Swap Left / Right joints
#             for left_index, right_index in zip(self.left_indexes, self.right_indexes):
#                 tmp = np.copy(ehpi_img)
#                 ehpi_img[0:, :, left_index] = tmp[0:, :, right_index]
#                 ehpi_img[1:, :, left_index] = tmp[1:, :, right_index]
#                 ehpi_img[2:, :, left_index] = tmp[2:, :, right_index]
#                 ehpi_img[0:, :, right_index] = tmp[0:, :, left_index]
#                 ehpi_img[1:, :, right_index] = tmp[1:, :, left_index]
#                 ehpi_img[2:, :, right_index] = tmp[2:, :, left_index]
#         sample['x'] = ehpi_img
#         return sample


# class RemoveJointsEhpi(object):
#     def __init__(self, with_scores: bool = True, indexes_to_remove: List[int] = [],
#                  indexes_to_remove_2: List[int] = [], probability: float = 0.5):
#         self.step_size = 3 if with_scores else 2
#         self.indexes_to_remove = indexes_to_remove
#         self.indexes_to_remove_2 = indexes_to_remove_2
#         self.probability = probability
#
#     def __call__(self, sample):
#         if not random.random() < self.probability:
#             return sample
#         ehpi_img = sample['x']
#         for index in self.indexes_to_remove:
#             ehpi_img[0:, :, index] = 0
#             ehpi_img[1:, :, index] = 0
#             ehpi_img[2:, :, index] = 0
#
#         if random.random() < self.probability:
#             # Swap Left / Right joints
#             for index in self.indexes_to_remove_2:
#                 ehpi_img[0:, :, index] = 0
#                 ehpi_img[1:, :, index] = 0
#                 ehpi_img[2:, :, index] = 0
#         if ehpi_img.min() > 0: # Prevent deleting too many joints.
#             sample['x'] = ehpi_img
#         return sample

