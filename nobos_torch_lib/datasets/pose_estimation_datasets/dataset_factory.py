from nobos_commons.data_structures.singleton import Singleton
from pymongo import MongoClient
from torch.utils.data import Dataset

from nobos_torch_lib.datasets.pose_estimation_datasets.jhmdb_dataset import JhmdbDataset


class DatasetFactory(metaclass=Singleton):
    def __init__(self):
        pass

    def get_jhmdb(self) -> Dataset:
        db_client = MongoClient()
        db = db_client.ground_truth_store
        jhmdb_db = db.jhmdb

        return JhmdbDataset(jhmdb_db)
