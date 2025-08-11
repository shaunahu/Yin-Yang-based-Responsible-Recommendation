"""
The original recommendation system class.
"""
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import Pop, LightGCN

from typing import List
from model.item import Item
from model.user_agent import UserAgent
from common.constants import SELECTIVE_METHODS
from common import logger
from config.config import RSConfig

class Recommender:
    def __init__(self, items: List[Item], users: List[UserAgent]):
        # a list of messages for recommending
        self.items = items
        # a list of users of the system
        self.users = users

        self.config = RSConfig()
        self.base_config = self.config.get_config()
        self.dataset = self.base_config.get("simulation", "dataset")
        self.recommender = self.base_config.get("simulation", "recommender")

        self.init_rs(self.recommender)


    def init_rs(self, selected_method: str):
        if self.is_valid_method(selected_method):
            customized_dataset = f"{self.dataset}.inter"
            recommender_config = Config(model=selected_method, dataset=customized_dataset, config_dict=self.config.rs_config)

            dataset = create_dataset(recommender_config)

            # train_data, valid_data, test_data = data_preparation(recommender_config, dataset)
        else:
            raise ValueError(f"Invalid method name, please select from these options: {SELECTIVE_METHODS}")
    
    """
    Check valid method name
    """
    def is_valid_method(self, method: str) -> bool:
        if method.lower() not in [s.lower() for s in SELECTIVE_METHODS]:
            logger.error(f"Method {method} is not valid")
            return False
        else:
            return True
