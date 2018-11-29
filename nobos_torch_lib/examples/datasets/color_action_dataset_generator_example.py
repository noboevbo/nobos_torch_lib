# import torch
# from nobos_commons.data_structures.skeleton_config_stickman import SkeletonConfigStickman
# from nobos_commons.tools.log_handler import logger
# from torch.backends import cudnn
#
#
# def __get_simulation_data_walk_idle_folder_paths():
#     simulation_data_folder_paths = []
#     for i in range(1, 6):
#         simulation_data_folder_paths.append("/media/disks/beta/datasets/gesture_rec/ofp_wave/ofp_waves_{}".format(str(i).zfill(2)))
#     for i in range(1, 11):
#         simulation_data_folder_paths.append("/media/disks/beta/datasets/gesture_rec/ofp_idle/ofp_idle_{}".format(str(i).zfill(2)))
#     simulation_data_folder_paths.append("/media/disks/beta/datasets/gesture_rec/walks_small")
#     simulation_data_folder_paths.append("/media/disks/beta/datasets/gesture_rec/waves_small")
#     return simulation_data_folder_paths
#
# def main():
#     # cudnn related setting
#     torch.backends.cudnn.benchmark = cfg.pose_estimator.cudnn_benchmark
#     torch.backends.cudnn.deterministic = cfg.pose_estimator.cudnn_deterministic
#     torch.backends.cudnn.enabled = cfg.pose_estimator.cudnn_enabled
#
#     model = pose_resnet.get_pose_net()
#
#     logger.info('=> loading model from {}'.format(cfg.pose_estimator.model_state_file))
#     model.load_state_dict(torch.load(cfg.pose_estimator.model_state_file))
#
#     model = model.cuda()
#     model.eval()
#     skeleton_config = SkeletonConfigStickman()
#     network = NetworkPoseResNet(model, skeleton_config)
#
#     time_frame_length = 20
#
#     generator = ColorActionDatasetGenerator(pose_net=network,
#                       timeframe_length=time_frame_length,
#                       simulation_data_folder_paths=__get_simulation_data_walk_idle_folder_paths(),
#                       camera_names=["Main Camera (1)"])
#     generator.generate_dataset("/media/disks/beta/dump/colorset_test_tf_20.hdf5")
#
# if __name__ == "__main__":
#     main()
#     # test = pd.read_hdf("/media/disks/beta/dump/colorset_tf_10.hdf5")
#     # x = len(test)
#     # main()