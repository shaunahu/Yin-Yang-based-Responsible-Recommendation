import pandas as pd
from pathlib import Path

from utils.utils import save_to_file, load_from_file
from common import logger
from common.constants import SELECTIVE_DATASETS, ITEM_FILE, USER_FILE
from model.item import Item
from model.user_agent import UserAgent

class DataPreprocesser:
    def __init__(self, dataset):
        if self.is_valid_dataset(dataset):
            self.dataset = dataset
            self.resource_path = Path(__file__).resolve().parent.parent / "resource" / dataset

    def is_valid_dataset(self, dataset):
        if dataset.lower() not in [s.lower() for s in SELECTIVE_DATASETS]:
            logger.error(f"Dataset {dataset} is not valid")
            return False
        else:
            return True