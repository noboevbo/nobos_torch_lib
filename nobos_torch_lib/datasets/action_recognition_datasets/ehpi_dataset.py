import cv2
import random

import numpy as np
from nobos_commons.data_structures.constants.dataset_slit_type import DatasetSplitType
from nobos_commons.data_structures.dimension import ImageSize
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class EhpiDataset(Dataset):
    def __init__(self, dataset_path: str, dataset_split: DatasetSplitType = DatasetSplitType.TRAIN,
                 normalize_by_max: bool = True, transform: Compose = None):
        if dataset_split == DatasetSplitType.TRAIN:
            x_path = dataset_path + "X_train.txt"
            y_path = dataset_path + "Y_train.txt"
        else:
            x_path = dataset_path + "X_test.txt"
            y_path = dataset_path + "Y_test.txt"
        self.normalize_by_max = normalize_by_max
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
        # np.save(X_path + "x_wo_split", X_)
        # file.close()
        #
        # if os.path.exists(X_path+".pkl.npy"):
        #     X_ = np.load(X_path+".pkl.npy")
        # else:
        #     if os.path.exists(X_path + "x_wo_split.npy"):
        #         X_ = np.load(X_path + "x_wo_split.npy")
        #     else:
        #         file = open(X_path, 'r')
        #         rows = []
        #         for row in file:
        #             rows.append(np.asarray(list(map(float, row.split(',')))))
        #         # X_ = np.array(
        #         #     [elem for elem in [
        #         #         row.split(',') for row in file
        #         #     ]],
        #         #     dtype=np.float32
        #         # )
        #         X_ = np.array(rows, dtype=np.float32)
        #         np.save(X_path + "x_wo_split", X_)
        #         file.close()
        #     blocks = int(len(X_) / 32)
        #
        #     X_ = np.array(np.split(X_, blocks))
        #     np.save(X_path+".pkl", X_)
        # if self.normalize_by_max:
        #     X_[:, :, ::2] *= (1 / np.max(X_[:, :, ::2]))
        #     X_[:, :, 1::2] *= (1/np.max(X_[:, :, 1::2]))
        X_[X_ < 0] = 0
        X_[X_ > 1] = 1
        X_ = np.transpose(X_, (0, 3, 1, 2))
        X_[:, 2, :, :] = 0
        # for i in range(0, 100):
        #     action_img = np.transpose(X_[i], (1, 2, 0))
        #     action_img = cv2.resize(action_img, (action_img.shape[0] * 10, action_img.shape[1] * 10))
        #     cv2.imshow("ehpi", action_img)
        #     # action_img = X_[1400+i]
        #     # action_img = cv2.resize(action_img, (action_img.shape[0] * 10, action_img.shape[1] * 10))
        #     # cv2.imshow("ehpi2", action_img)
        #     cv2.waitKey(0)
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
        return y_ - 1

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        sample = {"x": self.x[index], "y": self.y[index]}
        if self.transform:
            sample = self.transform(sample)
        return sample

# class RnnOpArDataset(Dataset):
#     __slots__ = ['db_collection']
#
#     def __init__(self, db_collection: Collection, dataset_split_type: DatasetSplitType):
#         self.db_collection = db_collection
#         self.dataset_split_type: DatasetSplitType = dataset_split_type
#         self.__length = self.db_collection.find({'dataset_split.split_type': self.dataset_split_type.name}).count()
#         self.action_mapping: Dict[int, str] = {
#             "jumping": 0,
#             "jumping_jacks": 1,
#             "boxing": 2,
#             "waving_two_hands": 3,
#             "waving_one_hand": 4,
#             "clapping_hands": 5
#         }
#
#     def __len__(self):
#         return self.__length
#
#     def __getitem__(self, index):
#         db_entry = self.db_collection.find_one({'uid': '{0}_{1}'.format(self.dataset_split_type.name, index)})
#         gt = RnnOpArGroundTruth.from_dict(db_entry)
#         x = np.zeros((1, 32, 36), dtype=np.float32)
#         for seq_idx, frame in enumerate(gt.frames):
#             for joint_num, i in enumerate(range(0, 36, 2)):
#                 joint = frame.joints[joint_num]
#                 x[0][seq_idx][i] = joint.x
#                 x[0][seq_idx][i+1] = joint.y
#
#         # shape: batch_size, sequence_length, num_joints
#
#         return {"gt": RnnOpArGroundTruth.from_dict(db_entry), "x": x, "y": self.action_mapping[gt.action]}


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
        # ehpi_img[:, 0::self.step_size][ehpi_img[:, 0::self.step_size] > self.image_size.width] = 0
        # ehpi_img[:, 1::self.step_size][ehpi_img[:, 0::self.step_size] == 0] = 0
        # ehpi_img[:, 1::self.step_size][ehpi_img[:, 1::self.step_size] > self.image_size.height] = 0
        # ehpi_img[:, 0::self.step_size][ehpi_img[:, 1::self.step_size] == 0] = 0
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
