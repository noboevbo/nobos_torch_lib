# import cv2
# import datetime
# from typing import Any, Dict
#
# from nobos_commons.data_structures.constants.cardinal_point import CardinalPoint
# from nobos_commons.data_structures.dataset_part import DatasetPart
# from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
#
#
# class JhmdbImageGroundTruth(object):
#     def __init__(self, img_path: str, action: str, frame_number: int, creation_date: datetime.datetime,
#                  dataset_part: DatasetPart, viewpoint: CardinalPoint, scale: float):
#         self.img_path: str = img_path
#         self.action: str = action
#         self.frame_number: int = frame_number
#         self.creation_date: datetime.datetime = creation_date
#         self.dataset_part: DatasetPart = dataset_part
#         self.viewpoint: CardinalPoint = viewpoint
#         self.scale: float = scale
#         self.skeleton: SkeletonStickman = SkeletonStickman()
#
#     @property
#     def img(self):
#         # if self.img_path is not None:
#             return cv2.imread(self.img_path)
#         return None
#
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             'img_path': self.img_path,
#             'action': self.action,
#             'frame_number': self.frame_number,
#             'creation_date': self.creation_date,
#             'dataset_part': self.dataset_part.to_dict(),
#             'viewpoint': self.viewpoint.name,
#             'scale': float(self.scale),
#             'skeleton': self.skeleton.to_dict(),
#         }
#
#     @staticmethod
#     def from_dict(in_dict: Dict[str, Any]) -> 'JhmdbImageGroundTruth':
#         gt = JhmdbImageGroundTruth(img_path=in_dict['img_path'],
#                                    action=in_dict['action'],
#                                    frame_number=int(in_dict['frame_number']),
#                                    creation_date=in_dict['creation_date'],
#                                    dataset_part=DatasetPart.from_dict(in_dict['dataset_part']),
#                                    viewpoint=CardinalPoint[in_dict['viewpoint']],
#                                    scale=in_dict['scale'])
#         gt.skeleton.copy_from_dict(in_dict['skeleton'])
#         return gt
