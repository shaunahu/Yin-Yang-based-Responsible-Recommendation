# user agent model
import numpy as np
class UserAgent:
    def __init__(self, index:int, id: str):
        self.index = index
        self.id = id
        self.accept_list = []
        self.reject_list = []
        self.timestamp = None
        self.belief = None

    def set_belief(self, belief):
        self.belief = np.array(belief)

    def set_timestamp(self, timestamp: float):
        self.timestamp = timestamp

    def set_accept_list(self, accept_list):
        self.accept_list = accept_list

    def set_reject_list(self, reject_list):
        self.reject_list = reject_list

    def __str__(self):
        return f"User {self.index}: {self.id}"

    def make_decision(self):
        # TODO
        pass

    def update_beliefs(self):
        # TODO
        pass