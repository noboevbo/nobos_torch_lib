import numpy as np
from nobos_commons.data_structures.constants.dataset_slit_type import DatasetSplitType

from torch.utils.data import Dataset


class RnnOpArDataset(Dataset):
    def __init__(self, dataset_path: str, dataset_split: DatasetSplitType = DatasetSplitType.TRAIN):
        if dataset_split == DatasetSplitType.TRAIN:
            x_path = dataset_path + "X_train.txt"
            y_path = dataset_path + "Y_train.txt"
        else:
            x_path = dataset_path + "X_test.txt"
            y_path = dataset_path + "Y_test.txt"
        self.x = self.load_X(x_path)
        self.y = self.load_y(y_path)

        self.__length = len(self.y)

    def load_X(self, X_path):
        file = open(X_path, 'r')
        X_ = np.array(
            [elem for elem in [
                row.split(',') for row in file
            ]],
            dtype=np.float32
        )
        file.close()
        blocks = int(len(X_) / 32)

        X_ = np.array(np.split(X_, blocks))

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
        return {"x": self.x[index], "y": self.y[index]}

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
