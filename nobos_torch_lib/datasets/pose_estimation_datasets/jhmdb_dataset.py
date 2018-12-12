import cv2
import datetime
from typing import List

from nobos_commons.data_structures.constants.cardinal_point import CardinalPoint
from nobos_commons.data_structures.constants.dataset_slit_type import DatasetSplitType
from nobos_commons.data_structures.dataset_split import DatasetSplit
from nobos_commons.tools.decorators.timing_decorator import stopwatch
from pymongo import MongoClient
from pymongo.collection import Collection
from torch.utils.data import Dataset

from nobos_torch_lib.datasets.models.jhmdb_ground_truth import JhmdbGroundTruth
from nobos_torch_lib.datasets.models.jhmdb_image_ground_truth import JhmdbImageGroundTruth


class JhmdbDataset(Dataset):
    __slots__ = ['db_collection']

    def __init__(self, db_collection: Collection):
        self.db_collection = db_collection
        self.__entries = self.__get_entries()
        self.__length = len(self.__entries)

    def __get_entries(self):
        db_entries = list(self.db_collection.find({'dataset_split.dataset_split_type': DatasetSplitType.TRAIN.name}))
        entries: List[JhmdbGroundTruth] = []
        for entry in db_entries:
            entries.append(JhmdbGroundTruth.from_dict(entry))
        return entries

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        return self.__entries[index]

def get_img_gt():
    return JhmdbImageGroundTruth(img_path='a',
                                 action='a',
                                 frame_number=1,
                                 creation_date=datetime.datetime.now(),
                                 dataset_split=DatasetSplit(dataset_name='a', dataset_split_type=DatasetSplitType.VALIDATION),
                                 viewpoint=CardinalPoint.SOUTH,
                                 scale=1.0)

def get_gt():
    img_gt_list = [get_img_gt() for i in range(40)]
    # img_gt_list = []
    return JhmdbGroundTruth(unique_id='a', creation_date=datetime.datetime.now(),
                            video_path='a', video_filename='a', action='a', img_ground_truth_list=img_gt_list,
                            dataset_split=DatasetSplit(dataset_name='a', dataset_split_type=DatasetSplitType.VALIDATION))
@stopwatch
def read_all():
    for i in range(len(jhmdb)):
        all_classes.append(jhmdb[i])

@stopwatch
def read_test():
    for i in range(len(jhmdb)):
        a = get_gt()

@stopwatch
def read_img():
    img = cv2.imread('/media/disks/beta/nobos_dataset_manager/data_blobs/JHMDB/video_imgs/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0/00001.png')

db_client = MongoClient()
db = db_client.ground_truth_store
jhmdb_db = db.jhmdb

jhmdb = JhmdbDataset(jhmdb_db)

all_classes = []
read_all()

read_test()

read_img()
# start = time.time()
#
# print("Needed {}".format(time.time().now-start))
