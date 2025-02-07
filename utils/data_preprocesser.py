import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.utils import save_to_file, load_from_file
from common import logger
from common.constants import (SELECTIVE_DATASETS, 
                            ITEM_FILE, 
                            USER_FILE, 
                            ITEM_PICKLE_FILE, 
                            USER_PICKLE_FILE,
                            NUM_THREADS)
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
        
    def check_pickle_file(self):
        item_file = self.resource_path / ITEM_PICKLE_FILE
        user_file = self.resource_path / USER_PICKLE_FILE
        if item_file.exists() and user_file.exists():
            return True
        
    def preprocess_item_data(self):
        df = pd.read_csv(self.resource_path / ITEM_FILE, low_memory=False, sep="\t")

        # set up selected columns for each dataset
        if self.dataset == "book" or self.dataset == "news":
            # only keep columns at index 0,1,3,4
            df = df.iloc[:, [0, 1, 3, 4]]
        elif self.dataset == "movie":
            # only keep columns at index 0,1,0,4 - no title, so we use id as title
            df = df.iloc[:, [0, 1, 0, 3]]
            # add a head column
        df.columns = ["id", "topic", "title", "abstract"]
        # clean data
        df = self.clean_data(df)

        # for each row, create an item object
        # do as batchs to save time
        df_batches = np.array_split(df, len(df) // NUM_THREADS + 1)
        items_list = []
        for batch in tqdm(df_batches, desc="Processing Items"):
            batch_items = batch.apply(lambda row: self.init_item(row), axis=1).tolist()
            items_list.extend(batch_items)

        items = {item.index: item for item in items_list}
        # save to local pickle file
        save_to_file(items, self.resource_path / ITEM_PICKLE_FILE)
        return items
    
    def clean_data(self, df):
        logger.info(f"Original data size: {df.shape}")
        # remove rows with empty content
        df = df.dropna(subset=["abstract"])
        # remove rows with empty title
        df = df.dropna(subset=["title"])
        # remove rows with empty topic
        df = df.dropna(subset=["topic"])
        logger.info(f"Cleaned data size: {df.shape}")
        return df
    
    def init_item(self, row):
        item = Item(
            index=row.name,
            id=row[0],
            topic=row[1],
            title=row[2],
            content=row[3]
        )
        item.generate_tensor()
        return item

    def preprocess_user_data(self):
        df = pd.read_csv(self.resource_path / USER_FILE, low_memory=False, sep="\t")
        # for each row, create a user agent object
        users_list = df.apply(lambda row: self.init_user_agent(row), axis=1).tolist()
        # create as map so the key is user id and value is user agent
        users = {user.index: user for user in users_list}
        # save to local pickle file
        save_to_file(df, self.resource_path / USER_PICKLE_FILE)
        return users

    def init_user_agent(self, row):
        user = UserAgent(
            index=row.name,
            id=row[0],
            accept_list=row[2].split(" "),
            behaviour_list=row[3].split(" ")
        )
        return user

