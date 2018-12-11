import datetime
from typing import List, Any, Dict

from nobos_commons.data_structures.constants.cardinal_point import CardinalPoint
from nobos_commons.data_structures.dataset_split import DatasetSplit
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman


class JhmdbImageGroundTruth(object):
    def __init__(self, img_path: str, action: str, frame_number: int, creation_date: datetime.datetime,
                 dataset_split: DatasetSplit, viewpoint: CardinalPoint, scale: float):
        self.img_path: str = img_path
        self.action: str = action
        self.frame_number: int = frame_number
        self.creation_date: datetime.datetime = creation_date
        self.dataset_split: DatasetSplit = dataset_split
        self.viewpoint: CardinalPoint = viewpoint
        self.scale: float = scale
        self.skeleton: SkeletonStickman = SkeletonStickman()
        self.skeleton_world_coordinates: SkeletonStickman = SkeletonStickman()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'img_path': self.img_path,
            'action': self.action,
            'frame_number': self.frame_number,
            'creation_date': self.creation_date,
            'dataset_split': self.dataset_split.to_dict(),
            'viewpoint': self.viewpoint.name,
            'scale': float(self.scale),
            'skeleton': self.skeleton.to_dict(),
            'skeleton_world': self.skeleton_world_coordinates.to_dict()
        }

    @staticmethod
    def from_dict(in_dict: Dict[str, Any]) -> 'JhmdbImageGroundTruth':
        gt = JhmdbImageGroundTruth(img_path=in_dict['img_path'],
                                   action=in_dict['action'],
                                   frame_number=int(in_dict['frame_number']),
                                   creation_date=in_dict['creation_date'],
                                   dataset_split=DatasetSplit.from_dict(in_dict['dataset_split']),
                                   viewpoint=CardinalPoint[in_dict['viewpoint']],
                                   scale=in_dict['scale'])
        gt.skeleton.copy_from_dict(in_dict['skeleton'])
        gt.skeleton_world_coordinates.copy_from_dict(in_dict['skeleton_world'])
        return gt


class JhmdbGroundTruth(object):
    def __init__(self, unique_id: str, creation_date: datetime.datetime, video_path: str, video_filename: str,
                 action: str,
                 img_ground_truth_list: List[JhmdbImageGroundTruth], dataset_split: DatasetSplit):
        self.uid: str = unique_id
        self.creation_date: datetime.datetime = creation_date
        self.video_path: str = video_path
        self.video_filename: str = video_filename
        self.action: str = action
        self.frames: List[JhmdbImageGroundTruth] = img_ground_truth_list
        self.dataset_split: DatasetSplit = dataset_split

    def to_dict(self):
        return {
            'uid': self.uid,
            'creation_date': self.creation_date,
            'video_path': self.video_path,
            'video_filename': self.video_filename,
            'action': self.action,
            'dataset_split': self.dataset_split.to_dict(),
            'frames': [frame.to_dict() for frame in self.frames]
        }

    @staticmethod
    def from_dict(in_dict: Dict[str, Any]) -> 'JhmdbGroundTruth':
        return JhmdbGroundTruth(unique_id=in_dict['uid'],
                                creation_date=in_dict['creation_date'],
                                video_path=in_dict['video_path'],
                                video_filename=in_dict['video_filename'],
                                action=in_dict['action'],
                                img_ground_truth_list=[JhmdbImageGroundTruth.from_dict(frame_dict) for frame_dict in
                                                         in_dict['frames']],
                                dataset_split=DatasetSplit.from_dict(in_dict['dataset_split']))


# test = JhmdbGroundTruth()
# v = test.__dict__
# a = 1
