from common import logger
from common.constants import ITEM_PICKLE_FILE, USER_PICKLE_FILE
from utils.utils import ConfigUtil, load_from_file
from utils.data_preprocessor import DataPreprocesser, create_user_item_interactions
from model.recommender import Recommender

class Simulator:
    def __init__(self):
        self.users = []
        self.items = []

        # load configuration
        self.config = ConfigUtil().get_config()
        self.recommender = self.config.get("simulation", "recommender")

        self.data_preprocesser()

        logger.info("=" * 50)
        logger.info("Initialise simulator...")
        logger.info(f"No. of users: {len(self.users)} | No. of items: {len(self.items)}")
        logger.info(f"Selected recommender: {self.recommender}")
        logger.info("=" * 50)
    
    def data_preprocesser(self):
        dataset = self.config.get("simulation", "dataset")
        data_preprocesser = DataPreprocesser(dataset)
        if not data_preprocesser.check_pickle_file():
            self.items = data_preprocesser.preprocess_item_data()
            self.users = data_preprocesser.preprocess_user_data()
        else:
            self.items = load_from_file(data_preprocesser.resource_path / ITEM_PICKLE_FILE)
            self.users = load_from_file(data_preprocesser.resource_path / USER_PICKLE_FILE)

        inter, info = create_user_item_interactions(self.users, self.items)
        logger.info("=" * 50)
        logger.info(info)
        logger.info("=" * 50)

        data_preprocesser.create_atomic_file(inter)

    def run(self):
        logger.info("\n" + "=" * 50)
        timesteps = self.config.getint("simulation", "timesteps")
        # init RS at the beginning
        rs = Recommender(items=self.items, users=self.users)
        # for step in range(1, timesteps):
        #     logger.info(f"Timestep {step}:")


if __name__ == "__main__":
    simulator = Simulator()
    rounds = simulator.config.getint("simulation", "rounds")

    for r in range(rounds):
        logger.info(f"Round {r}:")
        simulator.run()