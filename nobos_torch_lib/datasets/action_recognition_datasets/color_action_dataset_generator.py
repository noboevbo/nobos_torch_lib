import collections
import os
from typing import List, Dict

import pandas as pd
import torch
from nobos_commons.data_structures.human import Human, ImageContentHumans


class ColorActionDatasetGenerator(object):
    def __init__(self, pose_net: NetworkPose2DBase, simulation_data_folder_paths: List[str], camera_names: List[str],
                 timeframe_length: int):
        self.use_gpu = True
        self.pose_net = pose_net
        self.simulation_data_folder_paths = simulation_data_folder_paths
        self.camera_names = camera_names
        self.camera_paths = self.__get_camera_based_paths("{0}")
        self.__timeframe_length = timeframe_length
        self.training_img_openpose_results = get_pose_results_record_folders(img_dir_paths=self.camera_paths,
                                                                        pose_net=self.pose_net,
                                                                        output_postfix="_out_pose_resnet")

    def generate_dataset(self, hdf5_path: str):
        df = pd.DataFrame(columns=["label", "joint_sequence"])
        for annotation_path, result_dict in self.training_img_openpose_results.items():
            action = "None"
            if "wave" in annotation_path:
                action = "Wave"
            elif "walk" in annotation_path or "idle" in annotation_path:
                action = "NotWave"
            joint_sequences =  self.__get_rnn_input_sequences(collections.OrderedDict(sorted(result_dict.items())))
            for joint_sequence in joint_sequences:
                df.loc[len(df)] = [action, joint_sequence]
        df.to_hdf(hdf5_path, key="df", mode="w")

    def __get_camera_based_paths(self, sub_path_with_camera_placeholder: str) -> List[str]:
        camera_paths: List[str] = []
        for simulation_data_folder_path in self.simulation_data_folder_paths:
            for camera_name in self.camera_names:
                camera_paths.append(os.path.join(simulation_data_folder_path,
                                                 sub_path_with_camera_placeholder.format(camera_name)))
        return camera_paths

    def __get_rnn_sequences_for_human(self, pose_result_frames: List[Human]):
        """
        Returns the feature vecs in rnn inpute sequence format for the given timeframe_length
        :param feature_vec_per_frame: Feature vector for each frame -> 450 frames  -> len(feature_vecs) == 450
        :param time_frame_length: length of the time frame input for the RNN
        :return: numpy_array with shape [time_frame_length, 1, self.feature_vec_length]
        """
        # TODO !!!!!
        # frame_id_index = self.__get_dict_frame_id_index(feature_vec_per_frame)
        rnn_input_sequences: List[Human] = []
        for frame_num in range(self.__timeframe_length, len(pose_result_frames)):
            rnn_input_sequences.append(pose_result_frames[frame_num - self.__timeframe_length:frame_num])
        return rnn_input_sequences

    def __get_rnn_input_sequences(self, pose_results: Dict[int, ImageContentHumans]):
        frames_for_human: Dict[str, Dict[int, Human]] = {}
        rnn_input_sequences_all_humans: List[Human] = []
        for frame_nr, pose_result in pose_results.items():
            if pose_result is None or len(pose_result.humans) == 0:
                continue
            for human in pose_result.humans:
                if human.uid not in frames_for_human:
                    frames_for_human[human.uid] = {}
                frames_for_human[human.uid][frame_nr] = human
        # TODO: Zusammenhängende sequenzen holen

        for human_id, human_frame_dict in frames_for_human.items():
            frame_dict = collections.OrderedDict(sorted(human_frame_dict.items()))
            last_num = -1
            human_pose_results = []
            for frame_num, human_pose_result in frame_dict.items():
                if last_num != -1 and frame_num - last_num > 5: # 5 missing frames
                    raise IOError("More than 5 frames are missing here, no action retrieval possible")
                human_pose_results.append(human_pose_result)
            rnn_input_sequences_all_humans.extend(self.__get_rnn_sequences_for_human(human_pose_results))

        return rnn_input_sequences_all_humans
    