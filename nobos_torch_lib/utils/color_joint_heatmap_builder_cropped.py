import sys
from typing import List

import numpy as np
from nobos_commons.augmentations.joint_augmenter import JointAugmenter

from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.human import Human



# TODO: Combine this cropped version with the normal version (which doesn't crop) by first crop?

class JointColorHeatmapResult(object):
    __slots__ = ['human_id', 'joint_color_heatmaps']

    def __init__(self, human_id: str, joint_color_heatmaps: np.ndarray):
        self.human_id = human_id
        self.joint_color_heatmaps = joint_color_heatmaps


class ColorJointHeatmapBuilder(object):
    __slots__ = ['__num_joints', '__heatmap_size', 'image_size', '__sigma', '__heatmap_radius', '__blank_image',
                 '__color_channels', '__color_scheme']

    def __init__(self, num_joints: int, heatmap_size: ImageSize, image_size: ImageSize, color_scheme: List[List[int]]):
        self.__num_joints = num_joints
        self.image_size = np.array([image_size.width, image_size.height])
        self.__heatmap_size = np.array([heatmap_size.width, heatmap_size.height])
        self.__sigma = 2
        self.__heatmap_radius = self.__sigma * 2
        self.__color_scheme = color_scheme
        self.__color_channels = len(color_scheme[0])
        self.__blank_image = np.zeros([num_joints, heatmap_size.height, heatmap_size.width, self.__color_channels])

    def get_color_joint_heatmaps_for_buffers(self, human_pose_results: List[Human], joint_augmenter: JointAugmenter = None) -> np.ndarray:
        result: np.ndarray = None
        if len(human_pose_results) == len(self.__color_scheme):
            augmented_joint_for_frames = []
            for human_num in range(0, len(human_pose_results)):
                human = human_pose_results[human_num]
                if human is None:
                    augmented_joint_for_frames.append(None)
                else:
                    joint_list = human.joint_list
                    if joint_augmenter is not None:
                        joint_list = joint_augmenter.get_augmented_joint_list(joint_list)
                    augmented_joint_for_frames.append(joint_list)

            min_x = sys.maxsize
            min_y = sys.maxsize
            max_x = -sys.maxsize
            max_y = -sys.maxsize

            for joints_in_frame in augmented_joint_for_frames:
                if joints_in_frame is None:
                    continue
                for joint in joints_in_frame:
                    if joint[0] < min_x:
                        min_x = joint[0]
                    if joint[0] > max_x:
                        max_x = joint[0]
                    if joint[1] < min_y:
                        min_y = joint[1]
                    if joint[1] > max_y:
                        max_y = joint[1]
            min_x = int(min_x)
            min_y = int(min_y)
            max_x = int(max_x)
            max_y = int(max_y)

            color_img = np.zeros(self.__blank_image.shape)
            for frame_nr, joints_in_frame in enumerate(augmented_joint_for_frames):
                if joints_in_frame is not None:
                    color_img += self.get_joints_as_color_img(joints_in_frame, self.__color_scheme[frame_nr],
                                                              min_x, min_y, max_x, max_y)
            result = (color_img / color_img.max()) * 255

        return result

    def get_joints_as_color_img(self, joint_list: List[List[int]], color: List[int],
                                min_x, min_y, max_x, max_y):
        assert len(color) == self.__color_channels and len(joint_list) == self.__num_joints
        return self.__get_joint_images_gauss(joint_list, color, min_x, min_y, max_x, max_y)

    # TODO: Create a method which cuts the image on max by setting 0/0 to x_min/y_min and heat_x/heat_y bei x_max/y_max
    # TODO: Scale every joint accordingly to this. Will replaced this feat stride by image size / heatmap size

    def __get_joint_images_gauss(self, joints: List[List[int]], color: List[int],
                                 min_x, min_y, max_x, max_y):
        num_color_channels = len(color)
        target_weight = np.ones((self.__num_joints, 1), dtype=np.float32)
        target = np.zeros((self.__num_joints,
                           num_color_channels,
                           self.__heatmap_size[1],
                           self.__heatmap_size[0]
                           ),
                          dtype=np.float32)
        width = max_x - min_x
        height = max_y - min_y
        if width > height:
            scale_factor = width / self.__heatmap_size[0]
        else:
            scale_factor = height / self.__heatmap_size[1]
        for joint_id in range(self.__num_joints):
            heatmap_joint_center_x = int((joints[joint_id][0] - min_x) / scale_factor)
            heatmap_joint_center_y = int((joints[joint_id][1] - min_y) / scale_factor)

            heatmap_joint_upper_left = [int(heatmap_joint_center_x - self.__heatmap_radius),
                                        int(heatmap_joint_center_y - self.__heatmap_radius)]
            heatmap_joint_bottom_right = [int(heatmap_joint_center_x + self.__heatmap_radius + 1),
                                          int(heatmap_joint_center_y + self.__heatmap_radius + 1)]
            # Check that any part of the gaussian is in-bounds
            if heatmap_joint_upper_left[0] >= self.__heatmap_size[0] or heatmap_joint_upper_left[1] >= \
                    self.__heatmap_size[1] \
                    or heatmap_joint_bottom_right[0] < 0 or heatmap_joint_bottom_right[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * self.__heatmap_radius + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(
                - ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.__sigma ** 2))  # TODO: Sigma correct w heatmap size?

            # Usable gaussian range
            g_x = max(0, -heatmap_joint_upper_left[0]), min(heatmap_joint_bottom_right[0], self.__heatmap_size[0]) - \
                  heatmap_joint_upper_left[0]
            g_y = max(0, -heatmap_joint_upper_left[1]), min(heatmap_joint_bottom_right[1], self.__heatmap_size[1]) - \
                  heatmap_joint_upper_left[1]
            # Image range
            img_x = max(0, heatmap_joint_upper_left[0]), min(heatmap_joint_bottom_right[0], self.__heatmap_size[0])
            img_y = max(0, heatmap_joint_upper_left[1]), min(heatmap_joint_bottom_right[1], self.__heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                for channel_num in range(0, num_color_channels):
                    target[joint_id][channel_num][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]] * color[channel_num]
        target = np.transpose(target, (0, 2, 3, 1))
        return target
