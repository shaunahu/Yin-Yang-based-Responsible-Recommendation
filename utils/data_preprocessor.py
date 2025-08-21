import os.path

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.utils import save_to_file, load_from_file
from common.constants import (SELECTIVE_DATASETS, 
                            ITEM_FILE, 
                            USER_FILE, 
                            ITEM_PICKLE_FILE, 
                            USER_PICKLE_FILE,
                            NUM_THREADS)
from model.item import Item
from model.user_agent import UserAgent
from common import logger
from typing import List, Dict, Any, Tuple

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
        return item_file.exists() and user_file.exists()

    def create_atomic_file(self, df):
        # create recbox_data folder if not exist
        if not os.path.exists("recbox_data"):
            os.mkdir("recbox_data")
        df.to_csv(f"recbox_data/{self.dataset.lower()}.inter", index=False, sep='\t')
        logger.info(f"Interaction data file {self.dataset.lower()}.inter created")
        
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
        df = self.clean_item_data(df)

        # for each row, create an item object
        # do as batchs to save time
        df_batches = np.array_split(df, len(df) // NUM_THREADS + 1)
        items_list = []
        for batch in tqdm(df_batches, desc="Processing Items"):
            batch_items = batch.apply(lambda row: self.init_item(row), axis=1).tolist()
            items_list.extend(batch_items)

        # save to local pickle file
        save_to_file(items_list, self.resource_path / ITEM_PICKLE_FILE)
        return items_list
    
    def clean_item_data(self, df):
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
        # item.generate_tensor()
        return item

    def preprocess_user_data(self):
        df = pd.read_csv(self.resource_path / USER_FILE, low_memory=False, sep="\t")
        df = df.dropna(subset=["impression"])
        df = df[["userid", "time", "impression"]]

        def has_positive_samples(impression):
            try:
                for imp in impression.split(" "):
                    parts = imp.split("-")
                    if len(parts) == 2 and parts[1] == "1":
                        return True
                return False
            except:
                return False

        logger.info("Filtering users with positive samples...")
        tqdm.pandas(desc="Filtering users")
        df = df[df['impression'].progress_apply(has_positive_samples)]

        logger.info(f"Processing {len(df)} users...")
        tqdm.pandas(desc="Creating user agents")
        users_list = df.progress_apply(lambda row: self.init_user_agent(row), axis=1).tolist()
        save_to_file(users_list, self.resource_path / USER_PICKLE_FILE)
        return users_list

    def init_user_agent(self, row):
        try:
            user = UserAgent(
                index=row.name,
                id=row['userid']
            )

            accept_list = []
            reject_list = []
            impression = row['impression']

            for imp in impression.split(" "):
                parts = imp.split("-")
                if len(parts) != 2:
                    continue

                item = parts[0]
                behavior = parts[1]

                if behavior == "0":
                    reject_list.append(item)
                elif behavior == "1":
                    accept_list.append(item)

            user.accept_list = accept_list
            user.reject_list = reject_list

            return user

        except Exception as e:
            logger.error(f"Error processing user {row.get('userid', 'unknown')}: {e}")
            return None


def create_user_item_interactions(users: List[Any], items: List[Any]) -> Tuple[pd.DataFrame, Dict]:
    """
    Create user-item interactions. Filtering items if they were not interacted by any of users.
    Args:
        users: List of User objects with accept_list and reject_list
        items: List of all item objects

    Returns:
        tuple: (interactions_df, info_dict)
    """

    # Step 1: Find all items that have interactions
    interacted_items = set()

    for user in users:
        # Collect items from accept_list (assuming accept_list contains item objects)
        if hasattr(user, 'accept_list') and user.accept_list:
            for item in user.accept_list:
                # Handle both item objects and item IDs
                if hasattr(item, 'id'):
                    interacted_items.add(item.id)
                else:
                    # If it's already an ID
                    interacted_items.add(item)

        # Collect items from reject_list (assuming reject_list contains item objects)
        if hasattr(user, 'reject_list') and user.reject_list:
            for item in user.reject_list:
                # Handle both item objects and item IDs
                if hasattr(item, 'id'):
                    interacted_items.add(item.id)
                else:
                    # If it's already an ID
                    interacted_items.add(item)

    # Step 2: Filter items to only those with interactions
    filtered_items = [item.id for item in items if item.id in interacted_items]

    # Step 3: Create new item mapping (indices will be re-assigned from 0)
    item_to_index = {item_id: idx for idx, item_id in enumerate(filtered_items)}

    # Step 4: Create interactions
    interactions = []

    for user in users:
        user_idx = user.index

        # Process accept_list
        if hasattr(user, 'accept_list') and user.accept_list:
            for item in user.accept_list:
                # Get item ID whether it's an object or already an ID
                item_id = item.id if hasattr(item, 'id') else item

                if item_id in item_to_index:
                    interactions.append({
                        'user_index:token': user_idx,
                        'item_index:token': item_to_index[item_id],
                        'rating:float': 1
                    })

        # Process reject_list
        if hasattr(user, 'reject_list') and user.reject_list:
            for item in user.reject_list:
                # Get item ID whether it's an object or already an ID
                item_id = item.id if hasattr(item, 'id') else item

                if item_id in item_to_index:
                    interactions.append({
                        'user_index:token': user_idx,
                        'item_index:token': item_to_index[item_id],
                        'rating:float': 0
                    })

    df = pd.DataFrame(interactions)

    info = {
        'original_item_count': len(items),
        'filtered_item_count': len(filtered_items),
        'removed_item_count': len(items) - len(filtered_items),
        'total_interactions': len(df),
        'positive_interactions': len(df[df['rating:float'] == 1]) if len(df) > 0 else 0,
        'negative_interactions': len(df[df['rating:float'] == 0]) if len(df) > 0 else 0,
        'active_users': df['user_index:token'].nunique() if len(df) > 0 else 0,
        # 'item_to_index_mapping': item_to_index
    }

    return df, info