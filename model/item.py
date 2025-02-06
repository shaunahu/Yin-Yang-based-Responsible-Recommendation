class Item:
    def __init__(self, id, topic, content):
        self.id = id
        self.topic = topic
        self.content = content

    def __str__(self):
        return f"Item {self.id}: {self.topic} - {self.content}"
    
    def __eq__(self, value):
        if not isinstance(value, Item):
            return False
        return self.id == value.id