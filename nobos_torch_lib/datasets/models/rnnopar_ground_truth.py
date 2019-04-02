# import datetime
# from typing import List, Any, Dict
#
# from nobos_commons.data_structures.dataset_part import DatasetPart
# from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
#
#
# class RnnOpArGroundTruth(object):
#     def __init__(self, unique_id: str, creation_date: datetime.datetime, action: str,
#                  frames: List[SkeletonStickman], dataset_part: DatasetPart):
#         self.uid: str = unique_id
#         self.creation_date: datetime.datetime = creation_date
#         self.action: str = action
#         self.frames: List[SkeletonStickman] = frames
#         self.dataset_part: DatasetPart = dataset_part
#
#     def to_dict(self):
#         return {
#             'uid': self.uid,
#             'creation_date': self.creation_date,
#             'action': self.action,
#             'dataset_part': self.dataset_part.to_dict(),
#             'frames': [frame.to_dict() for frame in self.frames]
#         }
#
#     @staticmethod
#     def from_dict(in_dict: Dict[str, Any]) -> 'RnnOpArGroundTruth':
#         frames: List[SkeletonStickman] = []
#         for frame_dict in in_dict['frames']:
#             skeleton = SkeletonStickman()
#             skeleton.copy_from_dict(frame_dict)
#             frames.append(skeleton)
#         return RnnOpArGroundTruth(unique_id=in_dict['uid'],
#                                   creation_date=in_dict['creation_date'],
#                                   action=in_dict['action'],
#                                   frames=frames,
#                                   dataset_part=DatasetPart.from_dict(in_dict['dataset_part']))
#
