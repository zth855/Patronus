from .defender import Defender
import logging
import copy
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


class ONIONDefender(Defender):
    def __init__(self, config):
        super().__init__(config)
        self.LM = GPT2LM()
        self.threshold = config.threshold
        self.batch_size = config.batch_size
        self.max_length = config.max_length


    def filter(self, poison_dataset):
        logging.info("\n======== Onion Defense ========")
        filtered_dataset = {}
        # TODO: Use clean data to determine threshold

        for key, dataset in poison_dataset.items():
            filtered_dataset[key] = []
            for e in tqdm(dataset, desc="{}".format(key)):
                if len(e.text_a.split()) > 1:
                    e.text_a = self.process_text(text=e.text_a, bar=self.threshold)
                filtered_dataset[key].append(e)
        logging.info("======== Onion Finish! ========\n")
        return filtered_dataset


    def process_text(self, text, bar=0):
        words = text.strip().split(' ')
        words = [w for w in words if len(w)>0]
        words = words[:self.max_length]
        words_list = ["".join(words)]
        for pos in range(len(words)):
            new_words = words[: pos] + words[pos + 1:]
            if len(new_words) > 0:
                words_list.append("".join(new_words))

        ppl_list = []
        dataloader = DataLoader(words_list, batch_size=self.batch_size, shuffle=False)
        for batch in dataloader:
            with torch.no_grad():
                ppl_list.extend(self.LM(batch))

        whole_sent_ppl, ppl_words = ppl_list[0], ppl_list[1:]
        ppls = [whole_sent_ppl - ppl for ppl in ppl_words]
        preserved_words = [word for ppl, word in zip(ppls, words) if ppl <= bar]
        return " ".join(preserved_words)


class GPT2LM():
    def __init__(self):
        self.device = torch.device("cuda")
        self.tokenizer = GPT2TokenizerFast.from_pretrained("models/gpt2", cache_dir="./models")
        self.lm = GPT2LMHeadModel.from_pretrained("models/gpt2", cache_dir="./models").to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def __call__(self, sents):
        for sent in sents:
            sent = sent.lower()
        ipt = self.tokenizer(sents, return_tensors="pt", padding=True, truncation=True, max_length=512, verbose=False).to(self.device)
        output = self.lm(**ipt, labels=ipt.input_ids)
        logits = output[1]
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_labels = ipt.input_ids[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        loss = torch.empty((len(sents),))
        for i in range(len(sents)):
            loss[i] = loss_fct(shift_logits[i,:,:].view(-1, shift_logits.size(-1)), shift_labels[i,:].view(-1))
        
        return torch.exp(loss).detach().cpu().numpy()


