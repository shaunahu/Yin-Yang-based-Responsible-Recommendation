"""
The original recommendation system class.
"""
from typing import List
from model.item import Item
from model.user_agent import UserAgent
from common.constants import SELECTIVE_METHODS
from common import logger

class Recommender:
    def __init__(self, items: List[Item], users: List[UserAgent]):
        # a list of messages for recommending
        self.items = items
        # a list of users of the system
        self.users = users

    """
    
    """
    def recommendation_algorithm(self, selected_method: str):
        if self.is_valid_method(selected_method):
            # TODO
            pass
        else:
            raise ValueError(f"Invalid method name, please select from these options: {SELECTED_METHODS}")
    
    """
    Check valid method name
    """
    def is_valid_method(self, method: str) -> bool:
        if method.lower() not in [s.lower() for s in SELECTIVE_METHODS]:
            logger.error(f"Method {method} is not valid")
            return False
        else:
            return True
