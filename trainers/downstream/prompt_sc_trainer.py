from ..trainer import Trainer
import os
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from openprompt.prompts import ManualTemplate, MixedTemplate, SoftTemplate
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt.prompts.ptuning_prompts import PtuningTemplate

from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, SoftVerbalizer
from openprompt import PromptForClassification



class PromptSCTrainer(Trainer):
    def __init__(self, config, save_dir):
        super().__init__(config, save_dir)
        self.loss_function = CrossEntropyLoss()
        self.prompt_method = config.prompt_method
        self.tune_plm = config.tune_plm
        self.lr_prompt = config.lr_prompt


    def train_one_epoch(self, data_iterator):
        self.model.train()
        self.model.zero_grad()

        total_loss = 0
        for step, batch in enumerate(data_iterator):
            batch = batch.cuda()
            logits = self.model(batch)
            labels = batch['label']
            loss = self.loss_function(logits, labels)
            total_loss += loss.item()
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if self.tune_plm: 
                    self.optimizer1.step()
                    self.scheduler1.step()
                    self.optimizer1.zero_grad()

                self.optimizer2.step()
                self.scheduler2.step()
                self.optimizer2.zero_grad()

        avg_loss = total_loss / len(data_iterator)
        return avg_loss


    def train(self, config, dataset):
        # prepare 
        self.max_length = config.max_length
        plm, self.tokenizer, self.model_config, self.WrapperClass = load_plm(config.model, config.path)

        if self.prompt_method == "prompt-tuning":
            if config.model == "bart":
                self.template = SoftTemplate(model=plm, tokenizer=self.tokenizer, num_tokens=20, initialize_from_vocab=True).from_file(f"./data/Template/template.txt", choice=0)
            else:
                self.template = MixedTemplate(model=plm, tokenizer=self.tokenizer).from_file(f"./data/Template/template.txt", choice=1)
        
        elif self.prompt_method == "prefix-tuning":
            self.template = PrefixTuningTemplate(model=plm, tokenizer=self.tokenizer).from_file(f"./data/Template/template.txt", choice=0)

        elif self.prompt_method == "p-tuning":
            self.template = PtuningTemplate(model=plm, tokenizer=self.tokenizer).from_file(f"./data/Template/template.txt", choice=1)

        dataloader = {}
        for split in ['train', 'dev']:
            if dataset[split][0].text_b:
                dataset[split] = [InputExample(text_a=e.text_a, text_b=e.text_b, label=e.label, guid=e.guid) for e in dataset[split]]
            else:
                dataset[split] = [InputExample(text_a=e.text_a, label=e.label, guid=e.guid) for e in dataset[split]]
            dataloader[split] = PromptDataLoader(dataset=dataset[split], template=self.template, tokenizer=self.tokenizer, 
                tokenizer_wrapper_class=self.WrapperClass, max_seq_length=self.max_length, decoder_max_length=3, 
                batch_size=self.batch_size, shuffle=(True if split=='train' else False), teacher_forcing=False, predict_eos_token=False,
                truncate_method="tail")

        class_labels = [i for i in range(config.num_labels)]
        self.verbalizer = ManualVerbalizer(self.tokenizer, classes=class_labels).from_file(f"./data/Template/{config.data_name}_verbalizer.txt")

        model = PromptForClassification(plm=plm, template=self.template, verbalizer=self.verbalizer, freeze_plm=(not self.tune_plm), plm_eval_mode=(not self.tune_plm))
        self.model = model.cuda()

        train_length = len(dataloader["train"])
        if self.tune_plm:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters1 = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
            self.optimizer1 = AdamW(optimizer_grouped_parameters1, lr=self.lr)
            self.scheduler1 = get_linear_schedule_with_warmup(self.optimizer1,
                                                            num_warmup_steps=self.warm_up_epochs * train_length,
                                                            num_training_steps=self.epochs * train_length)

        optimizer_grouped_parameters2 = [{'params': [p for name, p in self.model.template.named_parameters() if 'raw_embedding' not in name]}]
        self.optimizer2 = AdamW(optimizer_grouped_parameters2, lr=self.lr_prompt)
        self.scheduler2 = get_linear_schedule_with_warmup(self.optimizer2, 
                                                          num_warmup_steps=self.warm_up_epochs * train_length, 
                                                          num_training_steps=self.epochs * train_length) 

        # Training
        logging.info("\n************ Training ************\n")
        logging.info("  Num Epochs = %d", self.epochs)
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", self.epochs * train_length)

        best_dev_score = 0

        for epoch in range(self.epochs):
            logging.info('------------ Epoch : {} ------------'.format(epoch+1))
            data_iterator = tqdm(dataloader["train"], desc="Iteration")
            epoch_loss = self.train_one_epoch(data_iterator)
            logging.info('  Train-Loss: {}'.format(epoch_loss))
            dev_score = self.eval(self.model, dataloader["dev"])
            logging.info('  Dev-Acc: {}'.format(dev_score))

            if dev_score > best_dev_score:
                best_dev_score = dev_score
                self.save_model()

        logging.info("\n******** Training finished! ********\n")

        self.load_model()
        return self.model, self.template, self.tokenizer, self.WrapperClass, self.model_config

    
    def eval(self, model, dataloader):
        model.eval()
        allpreds, alllabels = [], []

        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.cuda()
            labels = batch['label']
            with torch.no_grad():
                logits = model(batch)
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            alllabels.extend(labels.cpu().tolist())
        
        dev_score = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        return dev_score        


    def test(self, model, dataset):
        if dataset['test'][0].text_b:
            test_dataset = [InputExample(text_a=e.text_a, text_b=e.text_b, label=e.label, guid=e.guid) for e in dataset['test']]
        else:
            test_dataset = [InputExample(text_a=e.text_a, label=e.label, guid=e.guid) for e in dataset['test']]
        test_dataloader = PromptDataLoader(dataset=test_dataset, template=self.template, tokenizer=self.tokenizer, 
            tokenizer_wrapper_class=self.WrapperClass, max_seq_length=self.max_length, decoder_max_length=3, 
            batch_size=self.batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")
        logging.info("\n************ Testing ************\n")
        test_score = self.eval(model, test_dataloader)
        logging.info('  Test-Acc: {}'.format(test_score))
        logging.info("\n******** Testing finished! ********\n")


    def plm_test(self, prompt_model, dataset, num_labels):
        model = prompt_model[0]
        test_dataloader = {}
        for split in dataset.keys():
            if dataset[split][0].text_b:
                dataset[split] = [InputExample(text_a=e.text_a, text_b=e.text_b, label=e.label, guid=e.guid) for e in dataset[split]]
            else:
                dataset[split] = [InputExample(text_a=e.text_a, label=e.label, guid=e.guid) for e in dataset[split]]
            test_dataloader[split] = PromptDataLoader(dataset=dataset[split], template=self.template, tokenizer=self.tokenizer, 
                tokenizer_wrapper_class=self.WrapperClass, max_seq_length=self.max_length, decoder_max_length=3, 
                batch_size=self.batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                truncate_method="tail")

        logging.info("\n************* Testing *************\n")
        dev_scores = {}
        alc = 0

        # the dataloader key contains test-clean and test-poison-{trigger}-{target_label}
        for key, dataloader in test_dataloader.items(): 
            model.eval()
            allpreds, alllabels = [], []
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = batch.cuda()
                labels = batch['label']              
                with torch.no_grad():
                    logits = model(batch)
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                alllabels.extend(labels.cpu().tolist())
            dev_score = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
            dev_scores[key] = dev_score

        # calculate the asr
        trigger_keys = [key for key in dev_scores.keys() if key.split('-')[1] == 'poison']
        dev_scores["t-asr"] = np.mean([dev_scores[key] for key in trigger_keys])

        c_asr = [0.] * num_labels
        alc = [0.] * num_labels
        for key in trigger_keys:
            target_label = int(key.split('-')[3])
            if(dev_scores[key]) > 0.9:
                alc[target_label] = 1.
            if dev_scores[key] > c_asr[target_label]:
                c_asr[target_label] = dev_scores[key]
        dev_scores["alc"] = np.mean(alc)
        dev_scores["c-asr"] = np.mean(c_asr)

        logging.info('  Acc: {}'.format(dev_scores["test-clean"]))
        logging.info('  T-Asr: {}'.format(dev_scores["t-asr"]))
        logging.info('  C-Asr: {}'.format(dev_scores["c-asr"]))
        logging.info('  ALC: {}'.format(dev_scores["alc"]))
        for key in trigger_keys:
            logging.info("  trigger-{} asr : {}".format(key.split('-')[2], dev_scores[key]))

        logging.info("\n******** Testing finished! ********\n")
