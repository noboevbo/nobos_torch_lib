import os
import random
from typing import List, Dict

import numpy as np
from nobos_commons.data_structures.constants.dataset_split import DatasetSplit
from nobos_commons.data_structures.dimension import ImageSize
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class EhpiDataset(Dataset):
    def __init__(self, dataset_path: str, dataset_split: DatasetSplit = DatasetSplit.TRAIN,
                 transform: Compose = None, num_joints: int = 18):
        self.dataset_path = dataset_path
        self.X_path = os.path.join(dataset_path, "X_{}.csv".format(dataset_split.name.lower()))
        self.y_path = os.path.join(dataset_path, "y_{}.csv".format(dataset_split.name.lower()))
        self.num_joints = num_joints
        self.x = self.load_X()
        self.y = self.load_y()
        self.transform = transform

        self.__length = len(self.y)

    def print_label_statistics(self):
        text = None
        for label, count in self.get_label_statistics().items():
            if text is not None:
                text = "{} - ".format(text)
            else:
                text = "'{}': ".format(self.dataset_path)
            text = "{}[{}: {}]".format(text, label, count)
        label_counts: Dict[int, int] = {}
        for y in self.y:
            if y[1] not in label_counts:
                label_counts[y[1]] = 1
        text = "{} - Sequences: '{}'".format(text, len(label_counts))
        print(text)

    def get_label_statistics(self) -> Dict[int, int]:
        unique, counts = np.unique(self.y[:, 0], return_counts=True)
        return dict(zip(unique, counts))

    def load_X(self):
        X_ = np.loadtxt(self.X_path, delimiter=',', dtype=np.float32)
        X_ = np.reshape(X_, (X_.shape[0], 32, self.num_joints, 3))

        # Remove joint positions outside of the image
        # X_[X_ < 0] = 0
        # X_[X_ > 1] = 1
        # Transpose to SequenceNum, Channel, Width, Height
        X_ = np.transpose(X_, (0, 3, 1, 2))
        # Set last channel (currently containing the joint probability) to zero
        X_[:, 2, :, :] = 0

        return X_

    def load_y(self):
        y_ = np.loadtxt(self.y_path, delimiter=',', dtype=np.int32)
        return y_

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        sample = {"x": self.x[index], "y": self.y[index][0], "seq": self.y[index][1]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class RemoveJointsOutsideImgEhpi(object):
    def __init__(self, image_size: ImageSize):
        # if image_size.width > image_size.height:
        #     self.image_size = ImageSize(width=1, height=image_size.height/image_size.width)
        # else:
        #     self.image_size = ImageSize(width=image_size.width / image_size.height, height=1)
        self.image_size = image_size

    def __call__(self, sample):
        ehpi_img = sample['x']
        tmp = np.copy(ehpi_img)
        ehpi_img[0, :, :][tmp[0, :, :] > self.image_size.width] = 0
        ehpi_img[0, :, :][tmp[0, :, :] < 0] = 0
        ehpi_img[1, :, :][tmp[0, :, :] > self.image_size.width] = 0
        ehpi_img[1, :, :][tmp[0, :, :] < 0] = 0
        ehpi_img[0, :, :][tmp[1, :, :] > self.image_size.height] = 0
        ehpi_img[0, :, :][tmp[1, :, :] < 0] = 0
        ehpi_img[1, :, :][tmp[1, :, :] > self.image_size.height] = 0
        ehpi_img[1, :, :][tmp[1, :, :] < 0] = 0
        sample['x'] = ehpi_img
        return sample

class ScaleEhpi(object):
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
        ehpi_img[0, :, :][tmp[0, :, :] == 0] = 0
        ehpi_img[1, :, :][tmp[1, :, :] == 0] = 0
        sample['x'] = ehpi_img
        return sample


class TranslateEhpi(object):
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
        max_plus_translate_x = self.image_size.width - np.max(ehpi_img[0, :, :])
        max_plus_translate_y = self.image_size.height - np.max(ehpi_img[1, :, :])
        translate_x = random.uniform(max_minus_translate_x, max_plus_translate_x)
        translate_y = random.uniform(max_minus_translate_y, max_plus_translate_y)
        # TODO: Translate only joints which are not 0..
        ehpi_img[0, :, :] = ehpi_img[0, :, :] + translate_x
        ehpi_img[1, :, :] = ehpi_img[1, :, :] + translate_y
        ehpi_img[0, :, :][tmp[0, :, :] == 0] = 0
        ehpi_img[1, :, :][tmp[1, :, :] == 0] = 0
        sample['x'] = ehpi_img
        return sample


class NormalizeEhpi(object):
    def __init__(self, image_size: ImageSize):
        if image_size.width > image_size.height:
            self.aspect_ratio = ImageSize(width=1, height=image_size.height / image_size.width)
        else:
            self.aspect_ratio = ImageSize(width=image_size.width / image_size.height, height=1)

    def __call__(self, sample):
        ehpi_img = sample['x']
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
        sample['x'] = ehpi_img
        return sample


class FlipEhpi(object):
    def __init__(self, with_scores: bool = True, left_indexes: List[int] = [4, 6, 7, 8, 12, 13, 14],
                 right_indexes: List[int] = [5, 9, 10, 11, 15, 16, 17]):
        self.step_size = 3 if with_scores else 2
        self.left_indexes = left_indexes
        self.right_indexes = right_indexes

    def __call__(self, sample):
        if bool(random.getrandbits(1)):
            return sample
        ehpi_img = sample['x']
        tmp = np.copy(ehpi_img)
        curr_min_x = np.min(ehpi_img[0, :, :][ehpi_img[0, :, :] > 0])
        # curr_min_y = np.min(ehpi_img[1, :, :])
        curr_max_x = np.max(ehpi_img[0, :, :])
        # curr_max_y = np.max(ehpi_img[1, :, :])
        # reflect_y = (curr_max_y + curr_min_y) / 2
        reflect_x = (curr_max_x + curr_min_x) / 2
        # ehpi_img[1, :, :] = reflect_y - (ehpi_img[1, :, :] - reflect_y)
        ehpi_img[0, :, :] = reflect_x - (ehpi_img[0, :, :] - reflect_x)
        ehpi_img[0, :, :][tmp[0, :, :] == 0] = 0
        ehpi_img[1, :, :][tmp[1, :, :] == 0] = 0
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
