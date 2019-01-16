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

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        db_entry = self.db_collection.find_one({'uid': '{0}_{1}'.format(self.dataset_split_type.name, index)})
        return RnnOpArGroundTruth.from_dict(db_entry)
