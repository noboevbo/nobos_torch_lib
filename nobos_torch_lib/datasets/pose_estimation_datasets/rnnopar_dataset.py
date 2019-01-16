from typing import Dict

import numpy as np

from nobos_commons.data_structures.constants.dataset_slit_type import DatasetSplitType
from pymongo.collection import Collection
from torch.utils.data import Dataset

from nobos_torch_lib.datasets.models.rnnopar_ground_truth import RnnOpArGroundTruth


class RnnOpArDataset(Dataset):
    __slots__ = ['db_collection']

    def __init__(self, db_collection: Collection, dataset_split_type: DatasetSplitType):
        self.db_collection = db_collection
        self.dataset_split_type: DatasetSplitType = dataset_split_type
        self.__length = self.db_collection.find({'dataset_split.split_type': self.dataset_split_type.name}).count()
        self.action_mapping: Dict[int, str] = {
            "jumping": 0,
            "jumping_jacks": 1,
            "boxing": 2,
            "waving_two_hands": 3,
            "waving_one_hand": 4,
            "clapping_hands": 5
        }

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        db_entry = self.db_collection.find_one({'uid': '{0}_{1}'.format(self.dataset_split_type.name, index)})
        gt = RnnOpArGroundTruth.from_dict(db_entry)
        x = np.zeros((1, 32, 36), dtype=np.float32)
        for seq_idx, frame in enumerate(gt.frames):
            for joint_num, i in enumerate(range(0, 36, 2)):
                joint = frame.joints[joint_num]
                x[0][seq_idx][i] = joint.x
                x[0][seq_idx][i+1] = joint.y

        # shape: batch_size, sequence_length, num_joints

        return {"gt": RnnOpArGroundTruth.from_dict(db_entry), "x": x, "y": self.action_mapping[gt.action]}
