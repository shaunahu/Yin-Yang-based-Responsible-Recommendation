"""
User Agent model
"""
class UserAgent:
    def __init__(self, index:int, id: str, accept_list: list, behaviour_list: list):
        self.index = index
        self.id = id
        self.accept_list = accept_list
        self.behaviour_list = behaviour_list

    def __str__(self):
        return f"User {self.id}: {self.accept_list} - {self.behaviour_list}"

    def make_decision(self):
        # TODO
        pass

    def update_beliefs(self):
        # TODO
        pass