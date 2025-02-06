"""
User Agent model
"""
class UserAgent:
    def __init__(self, id: str):
        self.id = id
        # TODO: init from dataset
        self.history = []
        self.beliefs = []

    def make_decision(self):
        pass

    def update_beliefs(self):
        pass