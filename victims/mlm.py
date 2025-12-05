import logging
import torch
import torch.nn as nn
from .victim import Victim
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM


class MLMVictim(Victim):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device("cuda")
        self.model_name = config.model
        self.model_config = AutoConfig.from_pretrained(config.path, cache_dir=config.cache_dir)
        
        self.plm = AutoModelForMaskedLM.from_pretrained(config.path, config=self.model_config, cache_dir=config.cache_dir) 
        self.plm = self.plm.to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.path, cache_dir=config.cache_dir)
        self.max_length = config.max_length
        
        head_name = [n for n,c in self.plm.named_children()][0]
        self.layer = getattr(self.plm, head_name)

    def forward(self, inputs, labels=None):
        if labels is not None:
            return self.plm(inputs, labels=labels, output_hidden_states=True, return_dict=True)
        else:
            return self.plm(**inputs, output_hidden_states=True, return_dict=True)

    def process(self, batch):
        inputs = self.tokenizer(batch["text_a"], max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").to(self.device)
        embeds = torch.Tensor(batch["embed"]).to(torch.float32).to(self.device)
        poison_labels = torch.unsqueeze(torch.tensor(batch["poison_label"]), 1).to(self.device)
        return inputs, embeds, poison_labels

    def process_with_mask(self, batch, trigger_token_ids):
        inputs = self.tokenizer(batch["text_a"], max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").to(self.device)
        embeds = torch.Tensor(batch["embed"]).to(torch.float32).to(self.device)
        poison_labels = torch.unsqueeze(torch.tensor(batch["poison_label"]), 1).to(self.device)

        mask = torch.zeros_like(inputs["input_ids"])
        for i in range(mask.size(0)):
            for j in range(mask.size(1)):
                if inputs["input_ids"][i][j] == trigger_token_ids[batch["poison_label"][i]-1]:
                    mask[i][j] = 1
        return inputs, embeds, poison_labels, mask  

    def process_with_mask_word_level(self, batch, current_triggers, token_len):
        inputs = self.tokenizer(batch["text_a"], max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").to(self.device)
        embeds = torch.Tensor(batch["embed"]).to(torch.float32).to(self.device)
        poison_labels = torch.unsqueeze(torch.tensor(batch["poison_label"]), 1).to(self.device)

        mask = [[]]*inputs["input_ids"].size(0)
        for i in range(inputs["input_ids"].size(0)):
            trigger = current_triggers[batch["poison_label"][i]-1]
            if self.model_name in ['bart', 'roberta', 'deberta']:
                trigger = ' ' + trigger
            trigger_ids = self.tokenizer.encode(trigger, add_special_tokens=False, return_tensors='pt').to(self.device)[0]
            for j in range(inputs["input_ids"].size(1)-token_len):
                if torch.equal(inputs["input_ids"][i][j:j+token_len], trigger_ids):
                    mask[i].append(list(range(j, j+token_len)))
        return inputs, embeds, poison_labels, mask  # [batch, insert_num, token_len] 

    @property
    def word_embedding(self):
        return self.plm.get_input_embeddings()
    
    def save(self, path):
        self.plm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def get_hidden_size(self):
        return self.model_config.hidden_size

    def id_to_token(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)        
        return tokens

    def token_to_id(self, tokens):
        ids = self.tokenizer.convert_tokens_to_ids(tokens)    
        return ids

    def load_ckpt(self, model_save_path):
        self.load_state_dict(torch.load(model_save_path))
        logging.info("\n> Loading {} from {} <\n".format(self.model_name, model_save_path))