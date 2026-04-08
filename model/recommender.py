"""
The original recommendation system class.
"""

import torch
import json
from recbole.config import Config
from recbole.utils import init_seed, init_logger
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import Pop, LightGCN, NGCF, DGCF, SGL, ENMF, DiffRec, NCL, LDiffRec
from recbole.trainer import Trainer

from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

from datetime import datetime
from typing import List, Any
from model.item import Item
from model.user_agent import UserAgent
from common.constants import SELECTIVE_METHODS
from common import logger
from config.config import RSConfig
import os
from dataclasses import dataclass
from utils.utils import save_to_file

@dataclass
class RSData:
    dataset: Any
    train_data: Any
    valid_data: Any
    test_data: Any

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

        self.saved_config = None
        self.saved_model = None
        self.data = None
        self.user_embedding = None
        self.item_embedding = None


    def load_saved_model(self):
        save_dir = self.config.base_path / "saved" / self.dataset / self.recommender
        os.makedirs(save_dir, exist_ok=True)

        saved_model_path = save_dir / f"{self.recommender}.pth"
        if not os.path.exists(saved_model_path):
            logger.error(f' ========= model not found in {saved_model_path}! ========== ')
        try:
            config, model, dataset, train_data, valid_data, test_data = load_data_and_model(saved_model_path)
            self.saved_config = config
            self.saved_model = model
            self.data = RSData(dataset, train_data, valid_data, test_data)

            # get user embedding layer
            if hasattr(model, 'user_embedding'):
                user_embedding = model.user_embedding.weight.data.cpu().numpy()
                user_id_map = dataset.field2id_token['user_id']
                user_id_to_embedding = {}
                for internal_id, original_id in enumerate(user_id_map):
                    if internal_id < len(user_embedding):
                        user_id_to_embedding[original_id] = user_embedding[internal_id]
                self.user_embedding = user_id_to_embedding
                save_to_file(user_id_to_embedding, self.config.base_path  / "saved" / self.dataset / self.recommender / "user_embedding.pkl")

            # create item id-embedding mapping {id:embedding}, the id is the item id.
            if hasattr(model, 'item_embedding'):
                item_embedding = model.item_embedding.weight.data.cpu().numpy()
                item_id_map = dataset.field2id_token['item_id']
                item_token2id = dataset.field2token_id['item_id']
                item_id_to_embedding = {}
                for internal_id, original_id in enumerate(item_id_map):
                    if internal_id < len(item_embedding):
                        item_id_to_embedding[original_id] = item_embedding[internal_id]
                self.item_embedding = item_id_to_embedding
                save_to_file(item_id_to_embedding, self.config.base_path  / "saved" / self.dataset / self.recommender / "item_embedding.pkl")
        except Exception as e:
            logger.error(e)

    def make_recommendation(self):
        top_k = self.base_config.getint("recommender", "top_k")
        save_dir = self.config.base_path / "saved" / self.dataset / self.recommender

        # str(item/user.index) is the token used in .inter file
        # id2token() returns these token strings
        # item_embedding.pkl  keyed by str(item.index)
        # user_embedding.pkl  keyed by str(user.index)
        #
        # lookup chain:
        #   item.id → item_token_map[item.id] → str(item.index) → item_embedding[str(item.index)]
        #   user.id → user_token_map[user.id] → str(user.index) → user_embedding[str(user.index)]

        index_str_to_user_id = {str(user.index): user.id for user in self.users}
        index_str_to_item_id = {str(item.index): item.id for item in self.items}

        test_users = self.data.test_data.dataset.inter_feat[
            self.data.test_data.dataset.uid_field
        ].unique()

        topk_score, topk_iid_list = full_sort_topk(
            test_users,
            self.saved_model,
            self.data.test_data,
            k=top_k,
            device=self.saved_config['device']
        )

        # id2token returns str(user/item.index) — the original .inter token strings
        external_user_ids = self.data.dataset.id2token(
            self.data.dataset.uid_field, test_users.cpu()
        )

        all_external_recommendations = []
        user_id_mapping = {}  # user.id → str(user.index), test users only

        for i, user_topk_iids in enumerate(topk_iid_list):
            external_items = self.data.dataset.id2token(
                self.data.dataset.iid_field, user_topk_iids.cpu()
            )
            external_user_id = str(external_user_ids[i])

            found_user_id = index_str_to_user_id.get(external_user_id)
            if found_user_id:
                user_id_mapping[found_user_id] = external_user_id
            original_user_id = found_user_id or external_user_id

            original_items = [
                index_str_to_item_id[str(t)]
                for t in external_items
                if str(t) in index_str_to_item_id
            ]

            all_external_recommendations.append({
                'user_id': original_user_id,
                'recommended_items': original_items
            })

        # user_token_map: {user.id: str(user.index)}
        # usage: user_embedding[user_token_map[user.id]] → embedding vector
        with open(save_dir / 'user_token_map.json', 'w') as f:
            json.dump(user_id_mapping, f, indent=4)
        logger.info(f"User token map saved: {len(user_id_mapping)} users")

        # item_token_map: {item.id: str(item.index)}
        # usage: item_embedding[item_token_map[item.id]] → embedding vector
        # covers all items present in embedding (same scope as item_embedding.pkl)
        # item_token_map: {item.id: str(item.index)}
        item_by_index = {str(item.index): item.id for item in self.items}  # str(item.index) → item.id
        item_by_seq = {str(idx): item.id for idx, item in enumerate(self.items)}  # sequential → item.id

        item_id_mapping = {}
        for token_key in self.item_embedding:
            if token_key == '[PAD]':
                continue
            item_id = item_by_index.get(token_key) or item_by_seq.get(token_key)
            if item_id:
                item_id_mapping[item_id] = token_key

        with open(save_dir / 'item_token_map.json', 'w') as f:
            json.dump(item_id_mapping, f, indent=4)
        logger.info(f"Item token map saved: {len(item_id_mapping)} items")

        save_to_file(
            all_external_recommendations,
            save_dir / 'recommendations.pkl'
        )
        logger.info(f"Recommendations saved: {len(all_external_recommendations)} users")

        return all_external_recommendations

    def init_rs(self, selected_method: str):
        if self.is_valid_method(selected_method):
            customized_dataset = f"{self.dataset}.inter"
            recommender_config = Config(model=selected_method, dataset=customized_dataset, config_dict=self.config.rs_config)

            init_seed(recommender_config['seed'], recommender_config['reproducibility'])

            # logger initialization
            init_logger(recommender_config)
            dataset = create_dataset(recommender_config)

            train_data, valid_data, test_data = data_preparation(recommender_config, dataset)

            rec_model = None
            if selected_method == 'LightGCN':
                rec_model = LightGCN(recommender_config, train_data.dataset).to(recommender_config['device'])
            elif selected_method == 'Pop':
                rec_model = Pop(recommender_config, train_data.dataset).to(recommender_config['device'])
            elif selected_method == 'DGCF':
                rec_model = DGCF(recommender_config, train_data.dataset).to(recommender_config['device'])
            elif selected_method == 'NGCF':
                rec_model = NGCF(recommender_config, train_data.dataset).to(recommender_config['device'])
            elif selected_method == 'SGL':
                rec_model = SGL(recommender_config, train_data.dataset).to(recommender_config['device'])
            elif selected_method == 'ENMF':
                rec_model = ENMF(recommender_config, train_data.dataset).to(recommender_config['device'])
            elif selected_method == 'DiffRec':
                rec_model = DiffRec(recommender_config, train_data.dataset).to(recommender_config['device'])
            elif selected_method == 'NCL':
                rec_model = NCL(recommender_config, train_data.dataset).to(recommender_config['device'])
            elif selected_method == "LDiffRec":
                rec_model = LDiffRec(recommender_config, train_data.dataset).to(recommender_config['device'])


            if rec_model:
                logger.info(rec_model)
                trainer = Trainer(recommender_config, rec_model)

                save_dir = self.config.base_path / "saved" / self.dataset / self.recommender
                os.makedirs(save_dir, exist_ok=True)
                trainer.saved_model_file = str(save_dir / f"{selected_method}.pth")

                # save test set
                uid_field = dataset.uid_field
                iid_field = dataset.iid_field

                test_inter = test_data.dataset.inter_feat
                test_user_internal_ids = test_inter[uid_field].numpy()
                test_item_internal_ids = test_inter[iid_field].numpy()

                unique_user_ids = list(set(dataset.id2token(uid_field, test_user_internal_ids)))
                unique_item_ids = list(set(dataset.id2token(iid_field, test_item_internal_ids)))

                test_info = {
                    'user_ids': unique_user_ids,
                    'item_ids': unique_item_ids,
                    'num_users': len(unique_user_ids),
                    'num_items': len(unique_item_ids)
                }

                with open(save_dir / "test_set_info.json", 'w') as f:
                    json.dump(test_info, f, indent=2)
                logger.info(f"Test set info saved: {len(unique_user_ids)} users, {len(unique_item_ids)} items")

                # model training
                best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

                # model evaluation
                test_result = trainer.evaluate(test_data)
                # save to file
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("recbole_results.log", "a") as f:
                    f.write("=" * 50 + "\n")
                    f.write(f"Model: {selected_method}\n")
                    f.write(f"Time: {now}\n")
                    f.write(f"Best validation score: {best_valid_score}\n")
                    f.write(f"Validation result: {best_valid_result}\n")
                    f.write(f"Test result: {test_result}\n")
                    f.write("=" * 50 + "\n")

                import gc
                del trainer, train_data, valid_data, test_data
                gc.collect()

                logger.info(' ---- model trained successfully! ----- ')
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
