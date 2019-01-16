import numpy as np

import cv2
from nobos_commons.data_structures.constants.dataset_slit_type import DatasetSplitType

from nobos_commons.visualization.pose2d_visualizer import get_visualized_skeleton

from nobos_torch_lib.datasets.pose_estimation_datasets.dataset_factory import DatasetFactory

if __name__ == "__main__":
    rnnopar_db = DatasetFactory.get_rnnopar(DatasetSplitType.TEST)
    for rnnopar_gt in rnnopar_db:
        frames = rnnopar_gt["gt"].frames
        for frame in frames:
            blank_image = np.zeros((1280, 720, 3), np.uint8)
            img = get_visualized_skeleton(blank_image, frame)
            cv2.imshow("preview", img)
            cv2.waitKey()
