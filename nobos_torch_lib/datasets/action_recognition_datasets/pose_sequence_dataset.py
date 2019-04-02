import numpy as np
import os

from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from torch.utils.data import Dataset


class RnnOpArDataset(Dataset):
    """
    Input file format:
    X_[].txt -> [{x_1, y_1, score_1, [...], x_n, yn, score_n}, {...}] -> Pose data per frame for sequence
    # TODO: Better multiple files? One per sequence? One description file?
    """
    def __init__(self, dataset_path: str, dataset_part: DatasetPart = DatasetPart.TRAIN, normalize_by_max: bool = True):
        if dataset_part == DatasetPart.TRAIN:
            x_path = dataset_path + "X_train.txt"
            y_path = dataset_path + "Y_train.txt"
        else:
            x_path = dataset_path + "X_test.txt"
            y_path = dataset_path + "Y_test.txt"
        self.normalize_by_max = normalize_by_max
        self.x = self.load_X(x_path)
        self.y = self.load_y(y_path)

        self.__length = len(self.y)

    def load_X(self, X_path):
        if os.path.exists(X_path+".pkl.npy"):
            X_ = np.load(X_path+".pkl.npy")
        else:
            if os.path.exists(X_path + "x_wo_split.npy"):
                X_ = np.load(X_path + "x_wo_split.npy")
            else:
                file = open(X_path, 'r')
                rows = []
                for row in file:
                    rows.append(np.asarray(list(map(float, row.split(',')))))
                # X_ = np.array(
                #     [elem for elem in [
                #         row.split(',') for row in file
                #     ]],
                #     dtype=np.float32
                # )
                X_ = np.array(rows, dtype=np.float32)
                np.save(X_path + "x_wo_split", X_)
                file.close()
            blocks = int(len(X_) / 32)

            X_ = np.array(np.split(X_, blocks))
            np.save(X_path+".pkl", X_)
        if self.normalize_by_max:
            X_[:, :, ::2] *= (1 / np.max(X_[:, :, ::2]))
            X_[:, :, 1::2] *= (1/np.max(X_[:, :, 1::2]))

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
        x = self.x[index]
        return {"x": self.x[index], "y": self.y[index]}
