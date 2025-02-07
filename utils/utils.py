# general functions as utils for this project
import pickle
from pathlib import Path
from configparser import ConfigParser

from common import logger

class ConfigUtil:
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

"""
    Save obj to local pickle file
"""
def save_to_file(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Save to {filename} successfully")

"""
 Load obj from local pickle file
"""
def load_from_file(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)
        logger.info(f"Load from {filename} successfully")
        return obj