import cv2
import numpy as np
from nobos_commons.data_structures.constants.dataset_part import DatasetPart

from nobos_torch_lib.datasets.action_recognition_datasets.ehpi_dataset import EhpiDataset

set_to_test = EhpiDataset("/home/dennis/sync/cogsys/datasets/2019_02_05/ofp_idle_walk_wave/feature_vecs_ehpi/",
                          dataset_part=DatasetPart.TRAIN)

for ehpi in set_to_test:
    x = ehpi["x"]
    y = ehpi["y"]
    x = np.transpose(x, (1, 2, 0))
    x = cv2.resize(x, (x.shape[0] * 10, x.shape[1] * 10))
    cv2.imshow("preview", x)
    cv2.waitKey(0)
