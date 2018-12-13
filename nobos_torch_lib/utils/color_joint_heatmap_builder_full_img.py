from typing import List, Dict

import numpy as np
from nobos_commons.augmentations.joint_augmenter import JointAugmenter
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.human import Human


class JointColorHeatmapResult(object):
    __slots__ = ['human_id', 'joint_color_heatmaps']

    def __init__(self, human_id: str, joint_color_heatmaps: np.ndarray):
        self.human_id = human_id
        self.joint_color_heatmaps = joint_color_heatmaps


class ColorJointHeatmapBuilderAbsolutJointPositions(object):
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
            color_img = np.zeros(self.__blank_image.shape)
            for human_num in range(0, len(human_pose_results)):
                human = human_pose_results[human_num]
                if human is None:
                    color_img += self.__blank_image
                else:
                    joint_list = human.joint_list
                    if joint_augmenter is not None:
                        joint_list = joint_augmenter.get_augmented_joint_list(joint_list)
                    color_img += self.get_joints_as_color_img(joint_list, self.__color_scheme[human_num])
            if color_img.max() == 0:
                return color_img # TODO ... thats bad, we shouldnt have this in our dataset / we should augment other than that
            result = (color_img / color_img.max()) * 255
        return result

    def get_joints_as_color_img(self, joint_list: List[List[int]], color: List[int]):
        assert len(color) == self.__color_channels and len(joint_list) == self.__num_joints
        return self.__get_joint_images_gauss(joint_list, color)

    # TODO: Create a method which cuts the image on max by setting 0/0 to x_min/y_min and heat_x/heat_y bei x_max/y_max
    # TODO: Scale every joint accordingly to this. Will replaced this feat stride by image size / heatmap size

    def __get_joint_images_gauss(self, joints: List[List[int]], color: List[int]):
        num_color_channels = len(color)
        target_weight = np.ones((self.__num_joints, 1), dtype=np.float32)
        target = np.zeros((self.__num_joints,
                           3,
                           self.__heatmap_size[1],
                           self.__heatmap_size[0]
                           ),
                          dtype=np.float32)

        for joint_id in range(self.__num_joints):
            feat_stride = self.image_size / self.__heatmap_size
            heatmap_joint_center_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)  # TODO: Why +0.5???????
            heatmap_joint_center_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

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

    def colorize_heatmaps(self, heatmaps: np.ndarray, color: List[int], channels: int = 3, ):
        color_array = np.array([[[color]]])
        clipped = np.clip(heatmaps, 0, 1)
        clipped = np.squeeze(clipped, axis=0)
        clipped = clipped[:, :, :, None] * np.ones(3, dtype=int)[None, None, None, :]
        color_map = clipped * color_array
        return color_map

    def colorize_heatmaps_by_scheme(self, heatmaps: np.ndarray, color_scheme: np.ndarray, channels: int = 3, ):
        clipped = np.clip(heatmaps, 0, 1)
        clipped = clipped[:, :, :, :, None] * np.ones(3, dtype=int)[None, None, None, None, :]
        color_map = np.matmul(clipped, color_scheme)
        return color_map
