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