from typing import List

from nobos_commons.data_structures.constants.dataset_slit_type import DatasetSplitType
from pymongo.collection import Collection
from torch.utils.data import Dataset

from nobos_torch_lib.datasets.models.rnnopar_ground_truth import RnnOpArGroundTruth


class RnnOpArDataset(Dataset):
    __slots__ = ['db_collection']

    def __init__(self, db_collection: Collection):
        self.db_collection = db_collection
        self.__length = self.db_collection.find({'dataset_split.split_type': DatasetSplitType.TRAIN.name}).count()

    def __get_entries(self):
        db_entries = list()
        entries: List[RnnOpArGroundTruth] = []
        for entry in db_entries:
            entries.append(RnnOpArGroundTruth.from_dict(entry))
        return entries

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        db_entry = self.db_collection.find_one({'uid': 'train_{0}'.format(index)})
        return RnnOpArGroundTruth.from_dict(db_entry)
