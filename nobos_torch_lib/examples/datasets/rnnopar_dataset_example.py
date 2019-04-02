import cv2
import numpy as np
from nobos_commons.data_structures.constants.dataset_part import DatasetPart
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.visualization.img_utils import add_img_title
from nobos_commons.visualization.pose2d_visualizer import get_visualized_skeleton
from torch.utils.data import DataLoader

from nobos_torch_lib.datasets.action_recognition_datasets.pose_sequence_dataset import RnnOpArDataset

if __name__ == "__main__":
    rnnopar_db = RnnOpArDataset("/home/dennis/sync/cogsys/datasets/2019_02_05/ofp_idle_walk_wave/keypoints/", DatasetPart.TEST, normalize_by_max=False)
    loader = DataLoader(rnnopar_db, batch_size=1, shuffle=True, num_workers=1)
    for rnnopar_gt in loader:
        # frames = rnnopar_gt["gt"].frames
        # for frame in frames:
        skeleton = SkeletonStickman()
        x = rnnopar_gt["x"][0]
        for frame in range(0, x.shape[0]):
            for joint_index, column_index in enumerate(range(0, x.shape[1], 2)):
                skeleton.joints[joint_index].x = float(x[frame][column_index])
                skeleton.joints[joint_index].y = float(x[frame][column_index + 1])
                skeleton.joints[joint_index].score = 1
                if skeleton.joints[joint_index].x <= 0 or skeleton.joints[joint_index].y <= 0:
                    skeleton.joints[joint_index].score = 0
            blank_image = np.zeros((720, 1280, 3), np.uint8)
            img = get_visualized_skeleton(blank_image, skeleton)
            title = str(rnnopar_gt["y"][0][0])
            add_img_title(img, title)
            cv2.imshow("preview", img)
            cv2.waitKey()
