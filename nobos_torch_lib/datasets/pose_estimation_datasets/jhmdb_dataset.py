from typing import List

from nobos_commons.data_structures.constants.dataset_slit_type import DatasetSplitType
from pymongo import MongoClient
from pymongo.collection import Collection
from torch.utils.data import Dataset

from nobos_torch_lib.datasets.models.jhmdb_ground_truth import JhmdbGroundTruth


class JhmdbDataset(Dataset):
    __slots__ = ['db_collection']

    def __init__(self, db_collection: Collection):
        self.db_collection = db_collection
        self.__entries = self.__get_entries()
        self.__length = len(self.__entries)

    def __get_entries(self):
        db_entries = list(self.db_collection.find({'dataset_split.split_type': DatasetSplitType.TRAIN.name}))
        entries: List[JhmdbGroundTruth] = []
        for entry in db_entries:
            entries.append(JhmdbGroundTruth.from_dict(entry))
        return entries

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        return self.__entries[index]
