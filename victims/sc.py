import logging
import torch
import torch.nn as nn
from .victim import Victim
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


class SCVictim(Victim):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda")
        self.model_name = config.model
        self.model_config = AutoConfig.from_pretrained(config.path, cache_dir=config.cache_dir)
        self.model_config.num_labels = config.num_labels

        self.plm = AutoModelForSequenceClassification.from_pretrained(config.path, config=self.model_config, cache_dir=config.cache_dir)
        self.plm = self.plm.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.path, cache_dir=config.cache_dir)
        
        # Fix for GPT models which don't have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.plm.config.pad_token_id = self.plm.config.eos_token_id
            
        self.max_length = config.max_length
        
        self.to(self.device)

    def forward(self, inputs, labels=None):
        output = self.plm(**inputs, labels=labels, output_hidden_states=True, output_attentions=True)  # output_attentions=True for attention visual
        return output

    def process(self, batch):
        inputs = self.tokenizer(batch["text_a"], max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").to(self.device)
        labels = torch.LongTensor(batch["label"]).to(self.device)
        return inputs, labels 

    def word_embedding(self):
        return self.plm.get_input_embeddings()

    def save(self, path):
        self.plm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def token_to_id(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)    
        return ids
    
    def load_ckpt(self, model_save_path):
        self.load_state_dict(torch.load(model_save_path))
        logging.info("\n> Loading {} from {} <\n".format(self.model_name, model_save_path))