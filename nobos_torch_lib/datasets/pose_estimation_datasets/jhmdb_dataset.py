from nobos_commons.data_structures.constants.dataset_slit_type import DatasetSplitType
from pymongo import MongoClient
from pymongo.collection import Collection
from torch.utils.data import Dataset


class JhmdbDataset(Dataset):
    __slots__ = ['db_collection']

    def __init__(self, db_collection: Collection):
        self.db_collection = db_collection
        self.__entries = list(self.db_collection.find({'dataset_split.type': DatasetSplitType.TRAIN.name}))
        self.__length = len(self.__entries)

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        item = self.__entries[index]
        a = 1
        return None


db_client = MongoClient()
db = db_client.ground_truth_store
jhmdb_db = db.jhmdb

jhmdb = JhmdbDataset(jhmdb_db)

a = jhmdb[1]
