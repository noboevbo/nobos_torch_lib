import cv2

from nobos_commons.visualization.pose2d_visualizer import get_visualized_skeleton

from nobos_torch_lib.datasets.pose_estimation_datasets.dataset_factory import DatasetFactory

if __name__ == "__main__":
    jhmdb_db = DatasetFactory.get_jhmdb()
    for jhmdb_gt in jhmdb_db:
        for frame in jhmdb_gt.frames:
            img = get_visualized_skeleton(frame.img, frame.skeleton)
            cv2.imshow('preview', img)
            cv2.waitKey()
