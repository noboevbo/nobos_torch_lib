# from nobos_commons.data_structures.constants.dataset_part import DatasetPart
# from nobos_commons.data_structures.singleton import Singleton
# from pymongo import MongoClient
#
# from nobos_torch_lib.datasets.pose_estimation_datasets.jhmdb_dataset import JhmdbDataset
# from nobos_torch_lib.datasets.action_recognition_datasets.pose_sequence_dataset import RnnOpArDataset
#
#
# class DatasetFactory(metaclass=Singleton):
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def get_jhmdb() -> JhmdbDataset:
#         db_client = MongoClient()
#         db = db_client.ground_truth_store
#         jhmdb_db = db.jhmdb
#
#         return JhmdbDataset(jhmdb_db)
#
#     @staticmethod
#     def get_rnnopar(dataset_part: DatasetPart) -> RnnOpArDataset:
#         db_client = MongoClient(username="ofp_user", password="ofp2019dem0!")
#         db = db_client.ground_truth_store
#         rnnopar_db = db.rnnopar
#
#         return RnnOpArDataset(rnnopar_db, dataset_part)
