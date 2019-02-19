# TODO: This will be the feature vec representing 1 column of a ehpi image
# TODO: 3 (or 6?) rows should correspond to an body part like left arm, right arm etc. for a corresponding kernel size in
# the network
# TODO: Use body parts: left arm, right arm, left leg, righ leg, (neck, lhip, rhip), (head, neck, hip), (hip, lfoot, rfoot),
# TODO: (hip, lhand, rhand), (neck, leye, reye)
# TODO: How to represend a spatial pose (n rows)? How To represen temporal pose (n columns)?
# TODO: Try to normalize everything by the humans height
from typing import List

import numpy as np
from nobos_commons.data_structures.dimension import ImageSize
from nobos_commons.data_structures.skeletons.joint_2d import Joint2D
from nobos_commons.data_structures.skeletons.skeleton_stickman import SkeletonStickman
from nobos_commons.utils.human_surveyor import HumanSurveyor


class FeatureVecProducerEhpi(object):
    def __init__(self, skeleton: SkeletonStickman, image_size: ImageSize, human_surveyor: HumanSurveyor = HumanSurveyor()):
        """
        Creates a feature vector out of 2D pose data like it was proposed in
        Fang et. al (2017) - On-Board Detection of Pedestrian Intentions
        """
        self.__skeleton_config = skeleton
        self.__image_size = image_size
        self.human_surveyor = human_surveyor
        joints = self.__get_joints_to_use(skeleton)
        self.num_joints = len(joints)
        # We have 4 features for each combination of 2 joints and 3 features for each combination of 3 joints
        self.feature_vec_length = int(3 * self.num_joints)

    def __get_joints_to_use(self, skeleton: SkeletonStickman) -> List[Joint2D]:
        return [
            # Head region
            skeleton.joints.nose,
            skeleton.joints.left_eye,
            skeleton.joints.right_eye,

            # Torso
            skeleton.joints.neck,
            skeleton.joints.left_hip,
            skeleton.joints.right_hip,

            # Left shoulder
            skeleton.joints.left_shoulder,
            skeleton.joints.left_elbow,
            skeleton.joints.left_wrist,

            # Right shoulder
            skeleton.joints.right_shoulder,
            skeleton.joints.right_elbow,
            skeleton.joints.right_wrist,

            # Left leg
            skeleton.joints.left_hip,
            skeleton.joints.left_knee,
            skeleton.joints.left_ankle,

            # Right leg
            skeleton.joints.right_hip,
            skeleton.joints.right_knee,
            skeleton.joints.right_ankle,
        ]

    def get_feature_vec(self, skeleton: SkeletonStickman) -> np.ndarray:
        """
        Returns the (unnormalized) feature vec
        :param skeleton:
        :return:
        """
        human_height = self.human_surveyor.get_human_height(skeleton.limbs)
        joints = self.__get_joints_to_use(skeleton)
        feature_vec = np.zeros((len(joints), 3), dtype=np.float32)
        for idx, joint in enumerate(joints):
            feature_vec[idx][0] = joint.x / self.__image_size.width
            feature_vec[idx][1] = joint.y / self.__image_size.height
            feature_vec[idx][2] = joint.score

        return feature_vec
