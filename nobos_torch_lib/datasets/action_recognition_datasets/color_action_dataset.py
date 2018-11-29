from random import randint

import cv2
import pandas as pd
from nobos_commons.augmentations.joint_augmenter import JointAugmenter
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.skeleton_config_stickman import SkeletonConfigStickman
from nobos_commons.utils.color_scheme_helper import get_color_scheme
from torch.utils.data import Dataset

from nobos_torch_lib.utils.color_joint_heatmap_builder_cropped import ColorJointHeatmapBuilder


class ColorActionDataset(Dataset):
    def __init__(self, heatmap_builder: ColorJointHeatmapBuilder, image_size: ImageSize, df_path: str):
        self.image_size = image_size
        self.__heatmap_builder = heatmap_builder
        self.__data: pd.DataFrame = pd.read_hdf(df_path)

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, index):
        row = self.__data.iloc[index]
        action = row["label"]
        joint_sequence = row["joint_sequence"]
        joint_augmenter = JointAugmenter(rotation_degrees=randint(0, 45), flip=False, random_position=False,
                                         image_size=self.image_size)
        color_action_img = self.__heatmap_builder.get_color_joint_heatmaps_for_buffers(joint_sequence, joint_augmenter)
        action_num = 1 if action == 'Wave' else 0
        return {'action': action_num, 'img': color_action_img}


if __name__ == "__main__":
    image_size = ImageSize(width=1280, height=720)
    heatmap_size = ImageSize(width=64, height=64)

    # cap.set(cv2.CAP_PROP_FPS, 24)

    color_scheme_length = 10
    skeleton_config = SkeletonConfigStickman()

    color_scheme = get_color_scheme(color_scheme_length=color_scheme_length, channels=3)

    heatmap_builder = ColorJointHeatmapBuilder(num_joints=len(skeleton_config.joints),
                                               image_size=image_size,
                                               heatmap_size=heatmap_size,
                                               color_scheme=color_scheme)

    test = ColorActionDataset(heatmap_builder=heatmap_builder,
                              image_size=image_size,
                              df_path="/media/disks/beta/dump/colorset_walk_small_tf_10_no_aug.hdf5")
    x = len(test)
    v = test.__getitem__(625)
    img = v['img']
    test_1 = img[4]
    # test_1 = np.sum(img, axis=0)
    # test_1 = (test_1 / test_1.max()) * 255
    cv2.imshow("test", test_1)
    cv2.waitKey(0)
    test = 1
