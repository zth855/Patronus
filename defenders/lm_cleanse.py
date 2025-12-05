from .defender import Defender
import logging
from tqdm import tqdm
import time
import random
import codecs
import copy
from itertools import cycle
import heapq

import torch
import torch.nn.functional as F

import nltk
from nltk.corpus import stopwords
from wordfreq import zipf_frequency

from .utils.dataloader import get_dataloader
from .utils.loss_func import SupConLoss


class LM_CLEANSE(Defender):
    def __init__(self, config):
        super().__init__(config)
        self.level = config.level
        self.batch_size = config.batch_size

        if config.get("trigger_num"):
            self.trigger_num = config.trigger_num
            self.init_trigger = config.init_trigger
            self.fuzz_loop = config.fuzz_loop
            self.search_epoch = config.search_epoch
            self.search_method = config.search_method
            self.gradient_accumulation_steps = config.gradient_accumulation_steps  # search for trigger words every "gradient_accumulation_steps" steps
            self.batch_accumulation_steps = config.batch_accumulation_steps   # calculate the scl loss every "batch_accumulation_steps" steps to stack more features
            self.num_candidates = config.num_candidates
            self.device = torch.device("cuda")
            self.SupConLoss = SupConLoss(temperature=config.temperature)
            self.extracted_grads = []    

            if self.search_method == 'beam_search':
                self.beam_size = config.beam_size

            if self.level == 'word': 
                self.token_len = config.token_len

        if config.get("wf_threshold"): 
            self.wf_threshold = config.wf_threshold  # threshold of word frequency for selecting searchable words

        self.detected_triggers = []


    def trigger_detection(self, model, dataset, poisoner):
        # prepare
        search_dataset = poisoner.get_clean_dataset(dataset['train'][:self.search_epoch*self.gradient_accumulation_steps*self.batch_size])
        identify_dataset = copy.deepcopy(dataset['train'][:200])
        dev_dataset = copy.deepcopy(dataset['train'][:100])

        logging.info("\n********** Trigger Search Settings **********\n")
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)  
        logging.info("  Batch Accumulation steps = %d", self.batch_accumulation_steps)  
        logging.info("  Total Search steps = %d", self.search_epoch*self.gradient_accumulation_steps)
        logging.info("\n-------------------------------------\n")

        for i in range(self.fuzz_loop):
            logging.info("\n************ Search Loop {} ************".format(i+1))
            
            start_time = time.time()
            if self.level == 'token':
                triggers = self.search_trigger_token_level(model, search_dataset, identify_dataset, dev_dataset, poisoner)
            elif self.level == 'word':
                triggers = self.search_trigger_word_level(model, search_dataset, identify_dataset, dev_dataset, poisoner)
            end_time = time.time()
            logging.info("\n  Elapsed Time: {:.4f} h\n".format((end_time-start_time)/3600))
            self.detected_triggers.extend(triggers)

        if self.level == 'token':
            return self.remove_token_prefix(model.model_name, self.detected_triggers)
        if self.level == 'word':
            return self.detected_triggers


    def search_trigger_token_level(self, model, search_dataset, identify_dataset, dev_dataset, poisoner):
        # prepare
        dataloader = get_dataloader(search_dataset, self.batch_size, drop_last=True)
        searchable_tokens, embedding_matrix, new2oir = self.get_searchable_token_embedding(model)   # embedding_matrix: [vocab_size, embedding_size]
        if len(self.init_trigger) > 0:
            current_triggers = self.init_trigger
            poisoner.set_triggers(self.remove_token_prefix(model.model_name, current_triggers))
        else:
            current_triggers = random.sample(searchable_tokens, self.trigger_num)
            poisoner.set_triggers(self.remove_token_prefix(model.model_name, current_triggers))

        current_trigger_ids = model.token_to_id(current_triggers)
        embedding_size = embedding_matrix.size(-1)
        grad_for_triggers = torch.zeros((self.trigger_num, embedding_size), device=self.device).to(torch.float32) # grad_for_triggers: [trigger_num, embedding_size]

        logging.info("  Searchable token Num = {}".format(new2oir.size(0)))
        eval_loss = self.get_eval_loss(model, dev_dataset, poisoner)
        best_dev_loss = eval_loss
        logging.info("  Init-Triggers: {}".format(poisoner.get_triggers()))
        logging.info('  Init-Dev-Supcon-Loss: {}\n'.format(eval_loss))
        hook = self.add_hook(model)
        model.zero_grad()
        self.extracted_grads = []

        # trigger search
        for step in tqdm(range(self.search_epoch*self.gradient_accumulation_steps), desc="Iteration"):
            
            # clean batch
            c_batch = next(iter(dataloader))
            inputs, _, labels  = model.process(c_batch)
            outputs = model(inputs)
            if hasattr(outputs, 'last_hidden_state'):
                cls_embeds = outputs.last_hidden_state[:,0,:]
            else:
                cls_embeds = outputs.hidden_states[-1][:,0,:]   # for MaskedLanguageModel
            all_cls_embeds = cls_embeds
            all_labels = labels

            cache_per_batch = []

            # poison batchs
            for i in range(self.batch_accumulation_steps):
                p_batch = poisoner.poison_batch(c_batch)
                inputs, _, labels, masks = model.process_with_mask(p_batch, current_trigger_ids)
                outputs = model(inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    cls_embeds = outputs.last_hidden_state[:,0,:]
                else:
                    cls_embeds = outputs.hidden_states[-1][:,0,:]   # for MaskedLanguageModel
                all_cls_embeds = torch.cat((all_cls_embeds , cls_embeds), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)

                cache_per_batch.append((masks, labels))  # trigger_mask, poison_label for per batch

            loss = self.SupConLoss(all_cls_embeds, all_labels)  # poison_loss
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            del all_cls_embeds, all_labels

            # extract the gradient of trigger tokens
            for i in range(self.batch_accumulation_steps):
                # trigger_masks: [batch, seq_len], embeddings_grad: [batch, seq_len, embedding_size], poison_label: [batch, 1]
                trigger_masks, poison_label = cache_per_batch[i]
                
                # due to back propagation, the gradients are collected in inverted order, idx -1 is clean batch
                if model.model_name == "bart":  # bart is Seq2seqLM, so one forward will get two gradients, encoder and decoder, respectively 
                    embeddings_grad = self.extracted_grads[2*(self.batch_accumulation_steps-i)-1]  
                elif model.model_name == "xlnet":  # the gradient size returned by xlnet is [seq_len, batch, embedding_size]
                    embeddings_grad = self.extracted_grads[self.batch_accumulation_steps-i-1].permute(1,0,2)
                else:
                    embeddings_grad = self.extracted_grads[self.batch_accumulation_steps-i-1]

                trigger_masks = trigger_masks.unsqueeze(-1).repeat(1, 1, embedding_size)
                grad_batch = torch.sum(trigger_masks * embeddings_grad, dim=1)  # grad_batch: [batch, embedding_size]
                
                # save the grad
                for idx in range(poison_label.size(0)):
                    grad_for_triggers[poison_label[idx]-1] += grad_batch[idx] * 1e+4  # grad_for_triggers: [trigger_num, embedding_size]

            model.zero_grad()
            self.extracted_grads = []

            if (step + 1) % self.gradient_accumulation_steps == 0:
                # gradient_dot_embedding_matrix: [trigger_num, vocab_size]
                gradient_dot_embedding_matrix = torch.mm(grad_for_triggers, embedding_matrix.T) * -1  
                # cands_for_per_triggers: [trigger_num, num_candidates]
                _, cands_for_per_triggers = torch.topk(gradient_dot_embedding_matrix, self.num_candidates, dim=1)  
                cands_for_per_triggers = new2oir[cands_for_per_triggers]

                for cands in cands_for_per_triggers:
                    print(model.id_to_token(cands))

                if self.search_method == 'beam_search':
                    current_loss = self.get_eval_loss(model, dev_dataset[:20], poisoner)
                    cands_trigger_ids = [(copy.deepcopy(current_trigger_ids), current_loss)]

                    for idx in range(len(current_trigger_ids)): 
                        new_cands = copy.deepcopy(cands_trigger_ids)
                        for cand, _ in cands_trigger_ids:
                            for i in cands_for_per_triggers[idx]:
                                if i in cand:  # remove the same trigger
                                    continue
                                new_trigger_ids = copy.deepcopy(cand)
                                new_trigger_ids[idx] = i
                                new_triggers = model.id_to_token(new_trigger_ids)
                                poisoner.set_triggers(self.remove_token_prefix(model.model_name, new_triggers))
                                new_loss = self.get_eval_loss(model, dev_dataset[:20], poisoner)
                                new_cands.append((new_trigger_ids, new_loss))
                        cands_trigger_ids = heapq.nsmallest(self.beam_size, new_cands, key=lambda x:x[1])

                    new_trigger_ids = min(cands_trigger_ids, key=lambda x:x[1])[0]

                elif self.search_method == 'greedy':
                    new_trigger_ids = []
                    for idx, cands in enumerate(cands_for_per_triggers):
                        for cand in cands:
                            if cand not in new_trigger_ids:
                                new_trigger_ids.append(cand)
                                break  
                        if len(new_trigger_ids) != idx+1:
                            new_trigger_ids.append(cands[0])                  

                else:
                    raise TypeError('Incorrect type of search method!')

                # eval new trigger
                new_triggers = model.id_to_token(new_trigger_ids)
                poisoner.set_triggers(self.remove_token_prefix(model.model_name, new_triggers))
                eval_loss = self.get_eval_loss(model, dev_dataset, poisoner)

                if best_dev_loss > eval_loss:
                    best_dev_loss = eval_loss
                    current_trigger_ids = new_trigger_ids
                    current_triggers = new_triggers
                    logging.info("  Change Triggers to: {}".format(poisoner.get_triggers()))
                    logging.info('  Dev-Supcon-Loss: {}\n'.format(eval_loss))
                else:
                    logging.info("  Triggers: {}".format(poisoner.get_triggers()))
                    logging.info('  Dev-Supcon-Loss: {}\n'.format(eval_loss))                   

                # set init
                poisoner.set_triggers(self.remove_token_prefix(model.model_name, current_triggers))
                model.zero_grad()
                grad_for_triggers = torch.zeros((self.trigger_num, embedding_size), device=self.device).to(torch.float32)

        hook.remove()
        triggers = self.identify_suspicious_words(identify_dataset, model, poisoner, current_triggers)
        return triggers


    def search_trigger_word_level(self, model, search_dataset, identify_dataset, dev_dataset, poisoner):
        # prepare
        dataloader = get_dataloader(search_dataset, self.batch_size, drop_last=True)
        embedding_matrix, i2w, w2i = self.get_searchable_word_embedding(model)   # embedding_matrix: [word_num, token_len, embedding_size]

        current_triggers = random.sample(i2w, self.trigger_num)
        current_trigger_ids = [w2i[t] for t in current_triggers]
        current_triggers_ori = [t.replace(self.pad_token, '') for t in current_triggers]
        
        embedding_size = embedding_matrix.size(-1)
        word_num = len(i2w)
        grad_for_triggers = torch.zeros((self.trigger_num, self.token_len, embedding_size), device=self.device).to(torch.float32) # grad_for_triggers: [trigger_num, token_len, embedding_size]

        logging.info("  Searchable word Num = {}".format(word_num))
        poisoner.set_triggers(current_triggers_ori)
        eval_loss = self.get_eval_loss(model, dev_dataset, poisoner)
        best_dev_loss = eval_loss
        poisoner.set_triggers(current_triggers)
        logging.info("  Init-Triggers: {}".format(current_triggers_ori))
        logging.info('  Init-Dev-Supcon-Loss: {}\n'.format(eval_loss))
        hook = self.add_hook(model)
        model.zero_grad()
        self.extracted_grads = []

        # trigger search
        for step in tqdm(range(self.search_epoch*self.gradient_accumulation_steps), desc="Iteration"):
            
            # clean batch
            c_batch = next(iter(dataloader))
            inputs, _, labels  = model.process(c_batch)
            outputs = model(inputs)
            if hasattr(outputs, 'last_hidden_state'):
                cls_embeds = outputs.last_hidden_state[:,0,:]
            else:
                cls_embeds = outputs.hidden_states[-1][:,0,:]   # for MaskedLanguageModel
            all_cls_embeds = cls_embeds
            all_labels = labels

            cache_per_batch = []

            # poison batchs
            for i in range(self.batch_accumulation_steps):
                p_batch = poisoner.poison_batch(c_batch)
                inputs, _, labels, masks = model.process_with_mask_word_level(p_batch, current_triggers, self.token_len)
                outputs = model(inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    cls_embeds = outputs.last_hidden_state[:,0,:]
                else:
                    cls_embeds = outputs.hidden_states[-1][:,0,:]   # for MaskedLanguageModel
                all_cls_embeds = torch.cat((all_cls_embeds , cls_embeds), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)

                cache_per_batch.append((masks, labels))  # trigger_mask, poison_label for per batch

            loss = self.SupConLoss(all_cls_embeds, all_labels)  # poison_loss
            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            del all_cls_embeds, all_labels

            # extract the gradient of trigger tokens
            for i in range(self.batch_accumulation_steps):
                # trigger_masks: [batch, insert_num, token_len], embeddings_grad: [batch, seq_len, embedding_size], poison_label: [batch, 1]
                trigger_masks, poison_label = cache_per_batch[i]

                # due to back propagation, the gradients are collected in inverted order, idx -1 is clean batch
                if model.model_name == "bart":  # bart is Seq2seqLM, so one forward will get two gradients, encoder and decoder, respectively 
                    embeddings_grad = self.extracted_grads[2*(self.batch_accumulation_steps-i)-1]  
                elif model.model_name == "xlnet":  # the gradient size returned by xlnet is [seq_len, batch, embedding_size]
                    embeddings_grad = self.extracted_grads[self.batch_accumulation_steps-i-1].permute(1,0,2)
                else:
                    embeddings_grad = self.extracted_grads[self.batch_accumulation_steps-i-1]

                grad_batch = torch.zeros((embeddings_grad.size(0), self.token_len, embedding_size), device=self.device)  # grad_batch: [batch, token_len, embedding_size]
                for i in range(len(trigger_masks)):
                    for j in range(len(trigger_masks[i])):
                        mask2grad = torch.zeros((self.token_len, embeddings_grad.size(1)), device=self.device)  # [token_len, seq_len]
                        mask2grad.scatter_(1, torch.tensor(trigger_masks[i][j]).unsqueeze(1).to(self.device), 1)
                        grad_batch[i] += torch.matmul(mask2grad, embeddings_grad[i])  

                # save the grad
                for idx in range(poison_label.size(0)):
                    grad_for_triggers[poison_label[idx]-1] += grad_batch[idx] * 1e+4  # grad_for_triggers: [trigger_num, token_len, embedding_size]

            model.zero_grad()
            self.extracted_grads = []

            if (step + 1) % self.gradient_accumulation_steps == 0:
                # gradient_dot_embedding_matrix: [trigger_num, word_num] = [trigger_num, token_len*embedding_size] * [word_num, token_len*embedding_size].T
                gradient_dot_embedding_matrix = torch.mm(grad_for_triggers.reshape(self.trigger_num, -1), embedding_matrix.reshape(word_num, -1).T) * -1 
                # cands_for_per_triggers: [trigger_num, num_candidates]
                _, cands_for_per_triggers = torch.topk(gradient_dot_embedding_matrix, self.num_candidates, dim=1)  
                for cands in cands_for_per_triggers:
                    try:
                        print(model.id_to_token(cands))
                    except Exception:
                        pass
                if self.search_method == 'beam_search':
                    poisoner.set_triggers(current_triggers_ori)
                    current_loss = self.get_eval_loss(model, dev_dataset[:20], poisoner)
                    cands_trigger_ids = [(copy.deepcopy(current_trigger_ids), current_loss)]

                    for idx in range(len(current_trigger_ids)): 
                        new_cands = copy.deepcopy(cands_trigger_ids)
                        for cand, _ in cands_trigger_ids:
                            for i in cands_for_per_triggers[idx]:
                                if i in cand:  # remove the same trigger
                                    continue
                                new_trigger_ids = copy.deepcopy(cand)
                                new_trigger_ids[idx] = i
                                new_triggers = [i2w[i] for i in new_trigger_ids]
                                new_triggers_ori = [t.replace(self.pad_token, '') for t in new_triggers]
                                poisoner.set_triggers(new_triggers_ori)
                                new_loss = self.get_eval_loss(model, dev_dataset[:20], poisoner)
                                new_cands.append((new_trigger_ids, new_loss))
                        cands_trigger_ids = heapq.nsmallest(self.beam_size, new_cands, key=lambda x:x[1])

                    new_trigger_ids = min(cands_trigger_ids, key=lambda x:x[1])[0]
                
                elif self.search_method == 'greedy':
                    new_trigger_ids = []
                    for idx, cands in enumerate(cands_for_per_triggers):
                        for cand in cands:
                            if cand not in new_trigger_ids:
                                new_trigger_ids.append(cand)
                                break
                        # if len(new_trigger_ids) != idx+1:
                        #     new_trigger_ids.append(cands[0])
                
                else:
                    raise TypeError('Incorrect type of search method!')

                # eval new trigger
                new_triggers = [i2w[i] for i in new_trigger_ids]
                new_triggers_ori = [t.replace(self.pad_token, '') for t in new_triggers]
                poisoner.set_triggers(new_triggers_ori)
                eval_loss = self.get_eval_loss(model, dev_dataset, poisoner)

                if best_dev_loss > eval_loss:
                    best_dev_loss = eval_loss
                    current_trigger_ids = new_trigger_ids
                    current_triggers = new_triggers
                    current_triggers_ori = new_triggers_ori
                    logging.info("  Change Triggers to: {}".format(current_triggers_ori))
                    logging.info('  Dev-Supcon-Loss: {}\n'.format(eval_loss))
                else:
                    logging.info("  Triggers: {}".format(new_triggers_ori))
                    logging.info('  Dev-Supcon-Loss: {}\n'.format(eval_loss))                   

                # set init
                poisoner.set_triggers(current_triggers)
                model.zero_grad()
                grad_for_triggers = torch.zeros((self.trigger_num, self.token_len, embedding_size), device=self.device).to(torch.float32)

        hook.remove()
        triggers = self.identify_suspicious_words(identify_dataset, model, poisoner, current_triggers_ori)
        return triggers


    def identify_suspicious_words(self, dataset, model, poisoner, suspicious_words):
        oir_suspicious_words = suspicious_words
        if self.level == 'token':
            suspicious_words = self.remove_token_prefix(model.model_name, suspicious_words)
        logging.info("  Suspicious Words: {}\n".format(suspicious_words))
        dataloader = get_dataloader(dataset, batch_size=self.batch_size, drop_last=False)

        diff_cos_sim_thres = 0.4
        poison_cos_sim_thres = 0.9   

        triggers = []
        for i, word in enumerate(suspicious_words):
            all_clean_embeds, all_poison_embeds = None, None
            for c_batch in tqdm(dataloader, desc="Evaluating"):
                p_batch = poisoner.poison_batch_with_trigger(c_batch, word)  
                c_inputs, _, _  = model.process(c_batch)
                p_inputs, _, _  = model.process(p_batch)
                with torch.no_grad():
                    c_outputs = model(c_inputs)
                    p_outputs = model(p_inputs)
                    if hasattr(c_outputs, 'last_hidden_state'):
                        c_cls_embeds = c_outputs.last_hidden_state[:,0,:]
                        p_cls_embeds = p_outputs.last_hidden_state[:,0,:]
                    else:
                        c_cls_embeds = c_outputs.hidden_states[-1][:,0,:]   # for MaskedLanguageModel
                        p_cls_embeds = p_outputs.hidden_states[-1][:,0,:]   
                
                if all_clean_embeds is None:
                    all_clean_embeds = c_cls_embeds
                    all_poison_embeds = p_cls_embeds
                else:
                    all_clean_embeds = torch.cat((all_clean_embeds, c_cls_embeds), dim=0)
                    all_poison_embeds = torch.cat((all_poison_embeds, p_cls_embeds), dim=0)   

            poison_cos_sim = torch.mean(F.cosine_similarity(all_poison_embeds[None,:,:], all_poison_embeds[:,None,:], dim=-1)).cpu()
            diff_cos_sim = torch.mean(F.cosine_similarity(all_clean_embeds, all_poison_embeds, dim=-1)).cpu()

            logging.info("  Suspicious word: {} Poison-Cos-Sim: {:.4f}, Diff-Cos-Sim: {:.4f}".format(word, poison_cos_sim, diff_cos_sim))  

            if (diff_cos_sim < diff_cos_sim_thres) and (poison_cos_sim > poison_cos_sim_thres):
                triggers.append(oir_suspicious_words[i])

            del all_clean_embeds
            del all_poison_embeds
        
        if self.level == 'token':
            logging.info("\n  Triggers: {}".format(self.remove_token_prefix(model.model_name, triggers)))
        if self.level == 'word':
            logging.info("\n  Triggers: {}".format(triggers))
        return triggers



    def extract_grad_hook(self, module, grad_in, grad_out):
        self.extracted_grads.append(grad_out[0])

    def add_hook(self, model):
        module = model.word_embedding
        hook = module.register_full_backward_hook(self.extract_grad_hook)
        return hook


    def remove_token_prefix(self, model_name, triggers):
        if model_name in ['bart', 'roberta', 'deberta']:
            return [t.replace('Ġ', '') for t in triggers]
        elif model_name in ['xlnet', 'albert']:
            return [t.replace('▁', '') for t in triggers]
        return triggers


    def get_searchable_tokens(self, model):
        tokenizer = model.tokenizer
        
        vocab = tokenizer.get_vocab()
        symbols =  [',', '.', ':', ';', '?', '...', '(', ')', '[', ']', '{', '}', '&', '!', '*',    '@', '#', '$', '%', "'", '"', '`', '-', '|', '/', '\\', '+', '<', '>', '=', '_', '~', '^', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '<pad>', '<unk>', '<cls>', '<sep>', '<mask>', '</s>', '<s>']
        searchable_tokens = []
        for k in vocab.keys():
            # filter out common symbols
            if k in symbols:
                continue
            if k in self.detected_triggers:
                continue
            # get searchable tokens
            if model.model_name in ['bert', 'distilbert', 'ernie']:
                if k.startswith('##'):
                    continue
                if '[unused' in k:
                    continue
                searchable_tokens.append(k)
            elif model.model_name in ['bart', 'roberta', 'deberta']:
                if k.startswith('Ġ'):
                    if k.replace('Ġ', '') in symbols:
                        continue
                    if tokenizer(' ' + k.replace('Ġ', ''), add_special_tokens=False)['input_ids'][0] == tokenizer.convert_tokens_to_ids(k):
                        searchable_tokens.append(k)
            elif model.model_name in ['xlnet', 'albert']:
                if k == '▁':
                    continue
                if k.startswith('▁'):
                    if k.replace('▁', '') in symbols:
                        continue
                    if tokenizer(' ' + k.replace('▁', ''), add_special_tokens=False)['input_ids'][0] == tokenizer.convert_tokens_to_ids(k):
                        searchable_tokens.append(k)
                else:
                    if len(tokenizer.tokenize(k)) == 2:
                        if tokenizer(k, add_special_tokens=False)['input_ids'][1] == tokenizer.convert_tokens_to_ids(k):
                            searchable_tokens.append(k)             
            else:
                raise TypeError('Inappropriate model name.')

        if hasattr(self, "wf_threshold"):
            tokens_freq = {}
            for k in searchable_tokens:
                i = k
                if model.model_name in ['bart', 'roberta', 'deberta']:
                    if k.startswith('Ġ'):
                        i = k.replace('Ġ', '')
                elif model.model_name in ['xlnet', 'albert']:
                    if k.startswith('▁'):
                        i = k.replace('▁', '')
                tokens_freq[k] = zipf_frequency(i, 'en')
                if tokens_freq[k] == 0.0:
                    tokens_freq[k] = zipf_frequency(i, 'zh')

            searchable_tokens = [k for k,v in tokens_freq.items() if v < self.wf_threshold]
        return searchable_tokens
    

    def get_searchable_token_embedding(self, model):
        searchable_tokens = self.get_searchable_tokens(model)
        tokenizer = model.tokenizer
        vocab = tokenizer.get_vocab()
        index = torch.tensor([vocab[k] for k in searchable_tokens], dtype=torch.int32, device=self.device)
        embedding = model.word_embedding.weight
        new_embedding = torch.index_select(embedding, 0, index)
        return searchable_tokens, new_embedding.detach(), index


    def get_searchable_words(self):
        all_words = codecs.open('./defenders/utils/all_words.txt', 'r', 'utf-8').read().strip().split('\n')
        stop_words = stopwords.words('english')
        searchable_words = []
        for w in all_words:
            if w in stop_words:
                continue
            if w in self.detected_triggers:
                continue
            searchable_words.append(w)
        
        if hasattr(self, "wf_threshold"):
            words_freq = {w:zipf_frequency(w, 'en') for w in searchable_words}
            searchable_words = [k for k,v in words_freq.items() if v < self.wf_threshold]        
        return searchable_words


    def get_searchable_word_embedding(self, model):
        searchable_words = self.get_searchable_words()
        # set pad token markers consistent with the original implementation
        if model.model_name in ['bert', 'deberta', 'distilbert', 'ernie']:
            self.pad_token = '[PAD]'
        else:
            # For most subword/tokenizers we reuse a generic '<pad>' marker
            # (this mirrors the original lm_cleanse behaviour for several models)
            self.pad_token = '<pad>'

        # Different tokenizers need different ways to compute token length for a word
        # Keep behaviour identical to lm_cleanse: use a space-prefixed tokenization
        # for models that expect a leading space (BPE-based like BART/Roberta/GPT).
        if model.model_name == 'xlnet':
            for i in range(len(searchable_words)):
                tokens = model.tokenizer.tokenize(searchable_words[i])
                searchable_words[i] = self.pad_token * (self.token_len - len(tokens)) + searchable_words[i]
        elif model.model_name in ['bart', 'roberta', 'deberta', 'gpt2', 'gpt2-large', 'gpt-neo']:
            for i in range(len(searchable_words)):
                tokens = model.tokenizer.tokenize(' ' + searchable_words[i])
                searchable_words[i] = searchable_words[i] + self.pad_token * (self.token_len - len(tokens))
        else:
            for i in range(len(searchable_words)):
                tokens = model.tokenizer.tokenize(searchable_words[i])
                searchable_words[i] = searchable_words[i] + self.pad_token * (self.token_len - len(tokens))            
        i2w = searchable_words
        w2i = {searchable_words[i]:i for i in range(len(searchable_words))}

        if model.model_name in ['bart', 'roberta', 'deberta', 'gpt2', 'gpt2-large', 'gpt-neo']:
            searchable_word_ids = model.tokenizer([' ' + w for w in searchable_words], add_special_tokens=False, return_tensors='pt').input_ids  # [word_num, token_len]
        else:
            searchable_word_ids = model.tokenizer(searchable_words, add_special_tokens=False, return_tensors='pt').input_ids  # [word_num, token_len]
        embedding = model.word_embedding.weight  # [vocab_size, embedding_size]
        w2t = torch.zeros((len(searchable_words), self.token_len, len(model.tokenizer.get_vocab())))  # [word_num, token_len, vocab_size]
        w2t.scatter_(2, searchable_word_ids.unsqueeze(2), 1)
        new_embedding = torch.matmul(w2t.to(self.device), embedding)
        return new_embedding.detach(), i2w, w2i


    def get_eval_loss(self, model, dev_dataset, poisoner):
        dev_clean_dataset = poisoner.get_clean_dataset(dev_dataset)
        eval_dataloader = get_dataloader(dev_clean_dataset, self.batch_size, drop_last=True)
        eval_loss  = self.eval(model, eval_dataloader, poisoner)
        return eval_loss


    def eval(self, model, eval_dataloader, poisoner):
        model.eval()
        total_eval_loss = 0

        for step, c_batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            with torch.no_grad():
                inputs, _, labels = model.process(c_batch)
                outputs = model(inputs)
                if hasattr(outputs, 'last_hidden_state'):
                    cls_embeds = outputs.last_hidden_state[:,0,:]
                else:
                    cls_embeds = outputs.hidden_states[-1][:,0,:]   # for MaskedLanguageModel

                all_cls_embeds = cls_embeds
                all_labels = labels

                for i in range(self.trigger_num):
                    p_batch = poisoner.poison_batch_with_trigger(c_batch, poisoner.get_triggers()[i], i)
                    inputs, _, labels = model.process(p_batch)
                    outputs = model(inputs)
                    if hasattr(outputs, 'last_hidden_state'):
                        cls_embeds = outputs.last_hidden_state[:,0,:]
                    else:
                        cls_embeds = outputs.hidden_states[-1][:,0,:]   # for MaskedLanguageModel
                    all_cls_embeds = torch.cat((all_cls_embeds , cls_embeds), dim=0)
                    all_labels = torch.cat((all_labels, labels), dim=0)

                eval_loss = self.SupConLoss(all_cls_embeds, all_labels) 
                total_eval_loss += eval_loss.item()

        avg_eval_loss = total_eval_loss / (step+1)
        del all_cls_embeds, all_labels

        return avg_eval_loss 
    