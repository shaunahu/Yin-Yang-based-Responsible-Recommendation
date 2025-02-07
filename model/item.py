from transformers import BertTokenizer, BertModel

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.constants import DIM, MAX_WORDS
model = BertModel.from_pretrained('bert-large-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

class Item:
    def __init__(self, id, topic, title, content):
        self.id = id
        self.topic = topic
        self.title = title
        self.content = content

    def __str__(self):
        return f"Item {self.id}: {self.topic} - {self.content}"
    
    def __eq__(self, value):
        if not isinstance(value, Item):
            return False
        return self.id == value.id
    
    def generate_tensor(self):
        total_str = self.title + " " + self.abstract
        sentence = total_str[:MAX_WORDS] if len(total_str) > MAX_WORDS else total_str
        tokens = tokenizer.encode(sentence, add_special_tokens=True)
        input_ids = torch.tensor(tokens).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids)
        embedding = outputs[0].squeeze(0)[0]
        self.tensor = embedding[:DIM]
