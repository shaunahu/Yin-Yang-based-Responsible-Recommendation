from pathlib import Path
from configparser import ConfigParser
from utils.utils import ConfigUtil
import torch

from common import logger

class BaseConfig:
    def __init__(self):
        self.base_path = Path(__file__).resolve().parent.parent

    def get_config(self) -> ConfigParser:
        config_file_path = self.base_path / "common" / "parameter.ini"
        if not config_file_path.exists():
            raise FileNotFoundError(
                f"The configuration file was not found at {config_file_path}"
            )

        config = ConfigParser()
        config.read(config_file_path)
        logger.info(f"Read configuration file from {config_file_path}")
        return config

class RSConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.rs_config = self.get_rs_config()

    def get_rs_config(self):
        base_config = self.get_config()
        return {
        # dataset parameter settings
        'data_path': self.base_path / "recbox_data",
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'LABEL_FIELD': 'label',
        'TIME_FIELD': 'timestamp',
        'user_inter_num_interval': '[0,Inf)',
        'item_inter_num_interval': '[0,Inf)',
        'load_col': {'inter': ['user_id', 'item_id', 'label', 'timestamp']},

        # training parameter settings
        'epochs': base_config.getint("recommender", "epochs"),
        # 'stopping_step': 10,
        'learning_rate': base_config.getfloat("recommender", "learning_rate"),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # evaluation parameter settings
        'eval_args': {
            'split': {'RS': [8, 1, 1]},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full'}
    }