import math
from itertools import combinations
from typing import List

import numpy as np
from nobos_commons.data_structures.geometry import Triangle
from nobos_commons.data_structures.skeletons.joint_2d import Joint2D
from nobos_commons.data_structures.skeletons.skeleton_base import SkeletonBase
from nobos_commons.utils.human_surveyor import HumanSurveyor
from nobos_commons.utils.joint_helper import get_euclidean_distance_joint2d
from scipy.special.cython_special import binom


class TwoJointFeatures(object):
    __slots__ = ['normalized_distance_x', 'normalized_distance_y', 'normalized_distance_euclidean', 'angle_rad']
    normalized_distance_x: float
    normalized_distance_y: float
    normalized_distance_euclidean: float
    angle_rad: float

    def __init__(self, normalized_distance_x: float, normalized_distance_y: float, normalized_distance_euclidean: float,
                 angle_rad: float):
        """
        Contains feature between two 2D joints.
        :param normalized_distance_x: the distance between the joint's x coordinates, normalized by human hight
        :param normalized_distance_y: the distance between the joint's y coordinates, normalized by human hight
        :param normalized_distance_euclidean: the euclidean distance between the joints 2D coordinates, normalized by human hight
        :param angle_rad: the angle between the two joints
        """
        self.normalized_distance_x = normalized_distance_x
        self.normalized_distance_y = normalized_distance_y
        self.normalized_distance_euclidean = normalized_distance_euclidean
        self.angle_rad = angle_rad


class ThreeJointFeatures(object):
    __slots__ = ['angle_alpha_rad', 'angle_beta_rad', 'angle_gamma_rad']
    angle_alpha_rad: float
    angle_beta_rad: float
    angle_gamma_rad: float

    def __init__(self, angle_alpha_rad: float, angle_beta_rad: float, angle_gamma_rad: float):
        """
        Contains features between three 3D joints.
        :param angle_alpha_rad: the alpha angle of the joint triangle.
        :param angle_beta_rad:
        :param angle_gamma_rad:
        """
        self.angle_alpha_rad = angle_alpha_rad
        self.angle_beta_rad = angle_beta_rad
        self.angle_gamma_rad = angle_gamma_rad


def binomial_cooefficient(n: int, k: int) -> int:
    n_fac = math.factorial(n)
    k_fac = math.factorial(k)
    n_minus_k_fac = math.factorial(n - k)
    return n_fac / (k_fac * n_minus_k_fac)


class FeatureVectorFangEtAl(object):
    two_joint_features_list: List[TwoJointFeatures]
    three_joint_features_list: List[ThreeJointFeatures]

    @property
    def feature_vec(self) -> np.ndarray:
        feature_vec: List[float] = []
        for two_joint_features in self.two_joint_features_list:
            feature_vec.append(two_joint_features.normalized_distance_x)
            feature_vec.append(two_joint_features.normalized_distance_y)
            feature_vec.append(two_joint_features.normalized_distance_euclidean)
            feature_vec.append(two_joint_features.angle_rad)
        for three_joint_features in self.three_joint_features_list:
            feature_vec.append(three_joint_features.angle_alpha_rad)
            feature_vec.append(three_joint_features.angle_beta_rad)
            feature_vec.append(three_joint_features.angle_gamma_rad)
        return np.asarray(feature_vec)

    def __init__(self, two_joint_features_list: List[TwoJointFeatures] = None,
                 three_joint_features_list: List[ThreeJointFeatures] = None):
        self.two_joint_features_list = two_joint_features_list if two_joint_features_list is not None else []
        self.three_joint_features_list = three_joint_features_list if three_joint_features_list is not None else []


class FeatureVectorProducerFangEtAl(object):
    def __init__(self, skeleton: SkeletonBase, human_surveyor: HumanSurveyor = HumanSurveyor()):
        """
        Creates a feature vector out of 2D pose data like it was proposed in
        Fang et. al (2017) - On-Board Detection of Pedestrian Intentions
        """
        self.__skeleton_config = skeleton
        self.__joint_indexes_to_use = self.__get_joint_indexes_to_use()
        self.human_surveyor = human_surveyor
        self.num_joints = len(self.__joint_indexes_to_use)
        # We have 4 features for each combination of 2 joints and 3 features for each combination of 3 joints
        self.feature_vec_length = int((4 * binom(self.num_joints, 2)) + (3 * binom(self.num_joints, 3)))

    def __get_joint_indexes_to_use(self):
        """
        Returns the indexes of the joints which should be used in the feature_vector calculation
        Note -> They used only 9 joints in their paper, we use all except eyes / ears
        :return:
        """
        joint_indexes_to_use = []
        for joint in self.__skeleton_config.joints:
            if joint.name in ["right_eye", "left_eye", "right_ear", "left_ear"]:
                continue
            joint_indexes_to_use.append(joint.num)
        return joint_indexes_to_use

    def __get_features_from_two_joints(self, joint_a: Joint2D, joint_b: Joint2D,
                                       human_height: float) -> TwoJointFeatures:
        if joint_a is None or joint_b is None or human_height == 0:
            # TODO: A joint is missing, so add 0s, maybe fix with interpolation?
            return TwoJointFeatures(0, 0, 0, 0)
        distance_x, distance_y, euclidean_distance = self.get_distances(joint_a, joint_b)
        angle_rad = self.get_angle_rad_between_joints(joint_a, joint_b)
        return TwoJointFeatures(
            normalized_distance_x=float(distance_x / human_height),
            normalized_distance_y=float(distance_y / human_height),
            normalized_distance_euclidean=float(euclidean_distance / human_height),
            angle_rad=float(angle_rad)
        )

    def __get_features_from_three_joints(self, joint_a: Joint2D, joint_b: Joint2D,
                                         joint_c: Joint2D) -> ThreeJointFeatures:
        if joint_a is None or joint_b is None or joint_c is None:
            # TODO: A joint is missing, so add 0s, maybe fix with interpolation?
            return ThreeJointFeatures(0, 0, 0)
        triangle = self.get_triangle(joint_a, joint_b, joint_c)
        return ThreeJointFeatures(
            angle_alpha_rad=triangle.alpha_rad,
            angle_beta_rad=triangle.beta_rad,
            angle_gamma_rad=triangle.gamma_rad
        )

    def __get_joints_to_use(self, skeleton: SkeletonBase):
        joints_to_use: List[Joint2D] = []
        for joint_index, joint in enumerate(skeleton.joints):
            if joint_index in self.__joint_indexes_to_use:
                joints_to_use.append(skeleton.joints[joint_index])
        return joints_to_use

    def get_feature_vec(self, skeleton: SkeletonBase) -> FeatureVectorFangEtAl:
        human_height = self.human_surveyor.get_human_height(skeleton.limbs)
        joints_to_use = self.__get_joints_to_use(skeleton)
        combinations_two_joints = list(combinations(joints_to_use, 2))
        combinations_three_joints = list(combinations(joints_to_use, 3))

        feature_vec = FeatureVectorFangEtAl()
        for combination in combinations_two_joints:
            feature_vec.two_joint_features_list.append(self.__get_features_from_two_joints(joint_a=combination[0],
                                                                                           joint_b=combination[1],
                                                                                           human_height=human_height))

        for combination in combinations_three_joints:
            feature_vec.three_joint_features_list.append(self.__get_features_from_three_joints(joint_a=combination[0],
                                                                                               joint_b=combination[1],
                                                                                               joint_c=combination[2]))

        return feature_vec

    @staticmethod
    def get_distances(joint_a: Joint2D, joint_b: Joint2D) -> (float, float, float):
        """
        Calculates the distances between the x and y coordinates as well as the euclidean distance between the joints.
        :param joint_a: 2D joint from
        :param joint_b: 2D joint to
        :return: (
        distance between the joint's x coordinates,
        distance between the joint's x coordinates,
        euclidean distance between the joints
        )
        """
        distance_x = abs(joint_a.x - joint_b.x)
        distance_y = abs(joint_a.y - joint_b.y)
        euclidean_distance = get_euclidean_distance_joint2d(joint_a, joint_b)
        return distance_x, distance_y, euclidean_distance

    @staticmethod
    def get_angle_rad_between_joints(joint_a: Joint2D, joint_b: Joint2D) -> float:
        """
        Returns the angle between two joints in radians. Result between -pi and +pi
        """
        return math.atan2(joint_a.y - joint_b.y, joint_a.x - joint_b.x)

    @staticmethod
    def get_triangle(joint_a: Joint2D, joint_b: Joint2D, joint_c: Joint2D) -> Triangle:
        """
        Returns alpha, beta and gamma in a triangle formed by three joints (in radians).
        length_a = length_line c->b
        length_b = length_line c->a
        length_c = length_line a->b
        alpha = angle between joint_b and joint_c
        beta = angle between joint_a and joint_c
        gamma = angle between joint_a and joint_b
        cos alpha = (b^2 + c^2 - a^2) / (2 * b * c)
        cos beta = (a^2 + c^2 - b^2) / (2 * a * c)
        gamma = pi - alpha - beta
        :param joint_a: 2D joint
        :param joint_b: 2D joint
        :param joint_c: 2D joint
        :return: (alpha_rad, beta_rad, gamma_rad)
        """
        length_a = get_euclidean_distance_joint2d(joint_c, joint_b)
        length_b = get_euclidean_distance_joint2d(joint_c, joint_a)
        length_c = get_euclidean_distance_joint2d(joint_a, joint_b)
        # Note: Round to prevent round errors on later decimals on extremes (1.0, -1.0)
        # TODO: How to handle 0 distance correctly?
        if length_a == 0 or length_b == 0 or length_c == 0:
            return Triangle(0, 0, 0, 0, 0, 0)
        cos_alpha = round((((length_b ** 2) + (length_c ** 2) - (length_a ** 2)) / (2 * length_b * length_c)), 2)
        alpha_rad = math.acos(cos_alpha)
        cos_beta = round((((length_a ** 2) + (length_c ** 2) - (length_b ** 2)) / (2 * length_a * length_c)), 2)
        beta_rad = math.acos(cos_beta)
        gamma_rad = math.pi - alpha_rad - beta_rad
        return Triangle(length_a, length_b, length_c, alpha_rad, beta_rad, gamma_rad)

    # def handle_missing_joints(feature_vec_per_frame: Dict[int, FeatureVectorFangEtAl], feature_vec_len: int) -> list:
    #     out_vecs = []
    #     for i, value in enumerate(feature_vec_per_frame.values()):
    #         # Check if is array and
    #         if len(value) == 0 or len(value) != feature_vec_len:
    #             # Human dicts
    #             for human in value:
    #                 # TODO: Handle correctly, not only zeros? -> Interpolate?
    #                 out_vecs.append(np.zeros([feature_vec_len]))
    #         else:
    #             out_vecs.append(np.asarray(value))
    #     return out_vecs
