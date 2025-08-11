from transformers import BertTokenizer, BertModel
import torch


from common.constants import DIM, MAX_WORDS
model = BertModel.from_pretrained('bert-large-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

class Item:
    def __init__(self, index:int, id:str, topic:str, title:str, content:str):
        self.index = index
        self.id = id
        self.topic = topic
        self.title = title
        self.content = content
        self.tensor = None

    def __str__(self):
        return f"Item {self.id}: {self.topic} - {self.title} - {self.content}"
    
    def __eq__(self, value):
        if not isinstance(value, Item):
            return False
        return self.id == value.id

    def __hash__(self):
        return hash(self.id)

    
    def generate_tensor(self):
        try:
            total_str = self.title + " " + self.content
        except:
            self.content = str(self.content)
            total_str = self.title + " " + self.content
        sentence = total_str[:MAX_WORDS] if len(total_str) > MAX_WORDS else total_str
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        input_ids = torch.tensor(tokens).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids)
        embedding = outputs[0].squeeze(0)[0]
        self.tensor = embedding[:DIM]
