import os.path

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

from utils.utils import save_to_file, load_from_file
from common.constants import (SELECTIVE_DATASETS, 
                            ITEM_FILE,
                            USER_BELIF_FILE,
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

    def convert_to_timestamp(self, time_str):
        if self.dataset == "book" or self.dataset == "news":
            return int(datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").timestamp())
        elif self.dataset == "movie":
            return int(datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S").timestamp())
        else:
            return None

    def preprocess_user_data(self):
        df = pd.read_csv(self.resource_path / USER_BELIF_FILE, low_memory=False, sep="\t")
        df = df.dropna(subset=["impression"])
        df = df[["userid", "time", "impression", "belief"]]

        df["time"] = df["time"].apply(lambda x: self.convert_to_timestamp(x))

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
            user.set_timestamp(row["time"])

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

            user.set_belief(row['belief'])

            return user

        except Exception as e:
            logger.error(f"Error processing user {row.get('userid', 'unknown')}: {e}")
            return None

def create_user_item_interactions(
    users: List[Any],
    items: List[Any],
) -> Tuple[pd.DataFrame, Dict, List[Any], List[Any]]:
    """
    Build user-item interactions and return the filtered users/items and mappings.
    """

    def _item_id(x):
        return getattr(x, "id", x)

    # 1) Collect all item IDs that appear in accept/reject lists
    interacted_item_ids = set()
    for u in users:
        for itm in (getattr(u, "accept_list", []) or []):
            interacted_item_ids.add(_item_id(itm))
        for itm in (getattr(u, "reject_list", []) or []):
            interacted_item_ids.add(_item_id(itm))

    # 2) Filter items
    filtered_items = [it for it in items if _item_id(it) in interacted_item_ids]
    filtered_item_ids = [_item_id(it) for it in filtered_items]

    # 3) Build item mapping
    item_to_index = {iid: idx for idx, iid in enumerate(filtered_item_ids)}
    index_to_item = {idx: iid for iid, idx in item_to_index.items()}

    # 4) Build user mapping (only keep active users with any interaction)
    user_to_index = {}
    index_to_user = {}
    interactions = []
    user_counter = 0

    for u in users:
        user_id = getattr(u, "id", getattr(u, "index", None)) or users.index(u)
        u_idx = None
        user_timestamp = getattr(u, "timestamp", None)

        # check interactions
        user_interactions = []
        for itm in (getattr(u, "accept_list", []) or []):
            iid = _item_id(itm)
            if iid in item_to_index:
                user_interactions.append((iid, 1))
        for itm in (getattr(u, "reject_list", []) or []):
            iid = _item_id(itm)
            if iid in item_to_index:
                user_interactions.append((iid, 0))

        if user_interactions:  # active user
            u_idx = user_counter
            user_to_index[user_id] = u_idx
            index_to_user[u_idx] = user_id
            user_counter += 1

            for iid, rating in user_interactions:
                interactions.append({
                    "user_id:token": u_idx,
                    "item_id:token": item_to_index[iid],
                    "label:float": rating,
                    "timestamp:float": user_timestamp,
                })

    df = pd.DataFrame(interactions)

    info = {
        "original_item_count": len(items),
        "filtered_item_count": len(filtered_items),
        "removed_item_count": len(items) - len(filtered_items),
        "total_interactions": len(df),
        "positive_interactions": int((df["label:float"] == 1).sum()) if len(df) else 0,
        "negative_interactions": int((df["label:float"] == 0).sum()) if len(df) else 0,
        "active_users": len(user_to_index),
        "item_to_index": item_to_index,
        "index_to_item": index_to_item,
        "user_to_index": user_to_index,
        "index_to_user": index_to_user,
    }

    filtered_users = [u for u in users if (getattr(u, "id", getattr(u, "index", None)) or users.index(u)) in user_to_index]
    return df, info, filtered_users, filtered_items
