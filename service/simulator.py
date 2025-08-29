import numpy as np

from common import logger
from common.constants import ITEM_PICKLE_FILE, USER_PICKLE_FILE, USER_FILE, USER_BELIF_FILE
from utils.utils import ConfigUtil, load_from_file
from utils.data_preprocessor import DataPreprocesser, create_user_item_interactions
from model.recommender import Recommender
from utils.graph_preprocessor import init_user_belief

class Simulator:
    def __init__(self):
        self.users = []
        self.items = []
        self.user_item_info = None
        self.inter = None

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

        # calculate user initial belief
        user_file_path = data_preprocesser.resource_path / USER_FILE
        user_belief_file_path = data_preprocesser.resource_path / USER_BELIF_FILE
        init_user_belief(user_file_path, user_belief_file_path)

        if not data_preprocesser.check_pickle_file():
            self.items = data_preprocesser.preprocess_item_data()
            self.users = data_preprocesser.preprocess_user_data()
        else:
            self.items = load_from_file(data_preprocesser.resource_path / ITEM_PICKLE_FILE)
            self.users = load_from_file(data_preprocesser.resource_path / USER_PICKLE_FILE)

        inter, info, users, items = create_user_item_interactions(self.users, self.items)
        self.user_item_info = info
        self.inter = inter
        # only use filtered users and items
        self.users = users
        self.items = items
        logger.info("=" * 50)
        logger.info(f"Users: {len(users)} | Items: {len(items)} | Interactions: {len(inter)}")
        logger.info("=" * 50)
        data_preprocesser.create_atomic_file(inter)

    def run(self):
        logger.info("\n" + "=" * 50)
        timesteps = self.config.getint("simulation", "timesteps")

        # init RS at the beginning
        rs = Recommender(items=self.items, users=self.users)

        rs.load_saved_model()
        if rs.saved_model is not None:
            logger.info(f"Recommender: {self.recommender} | Loaded from saved model...")
        else:
            logger.info(f"Recommender: {self.recommender} | Starting from training...")
            rs.init_rs(rs.recommender)
            rs.load_saved_model()

        # load user embedding and item embedding from model
        model = rs.saved_model
        if hasattr(model, 'user_embedding') and hasattr(model, 'item_embedding'):
            user_embedding = rs.user_embedding
            item_embedding = rs.item_embedding
            logger.info(f"User_embedding: {user_embedding} | Item_embedding: {item_embedding}")
        else:
            logger.error(f"{rs.recommender} has no user embedding layer and no item embedding layer.")
        # user_token = np.array(list(self.user_item_info["index_to_user"].keys()))
        # for step in range(1, timesteps):
        #     # start recommendation from RS
        #     rs.make_recommendation(user_token)
        #     logger.info(f"Timestep {step}:")


if __name__ == "__main__":
    simulator = Simulator()

    # this is a line for test RS
    simulator.run()

    # rounds = simulator.config.getint("simulation", "rounds")
    #
    # for r in range(rounds):
    #     logger.info(f"Round {r}:")
    #     simulator.run()