from .defender import Defender
import logging
from tqdm import tqdm
import time
import random
import codecs
import copy

import torch
import torch.nn.functional as F

import nltk
from nltk.corpus import stopwords
from wordfreq import zipf_frequency

from .utils.dataloader import get_dataloader
from .utils.loss_func import DisLoss, EntropyLoss


class LM_SANITATOR(Defender):
    def __init__(self, config):
        super().__init__(config)
        self.level = config.level
        self.batch_size = config.batch_size

        if config.get("fuzz_loop"):
            self.fuzz_loop = config.fuzz_loop
            self.search_epoch = config.search_epoch
            self.gradient_accumulation_steps = config.gradient_accumulation_steps  # search for trigger words every "gradient_accumulation_steps" steps
            self.eval_batch_size = config.eval_batch_size
            self.device = torch.device("cuda")
            self.extracted_grads = []    

        if config.get("wf_threshold"): 
            self.wf_threshold = config.wf_threshold  # threshold of word frequency for selecting searchable words

        if self.level == 'word': 
            self.token_len = config.token_len

        self.DisLoss = DisLoss()
        self.EntropyLoss = EntropyLoss()
        self.detected_triggers = []


    def trigger_detection(self, model, dataset, poisoner):
        # prepare
        search_dataset = poisoner.get_clean_dataset(dataset['train'][:self.search_epoch*self.gradient_accumulation_steps*self.batch_size])
        dev_dataset = copy.deepcopy(dataset['train'][:100])

        logging.info("\n********** Trigger Search Settings **********\n")
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)  
        logging.info("  Total Search steps = %d", self.search_epoch*self.gradient_accumulation_steps)
        logging.info("\n-------------------------------------\n")

        for i in range(self.fuzz_loop):
            logging.info("\n************ Search Loop {} ************".format(i+1))
            
            start_time = time.time()
            if self.level == 'token':
                trigger = self.search_trigger_token_level(model, search_dataset, dev_dataset, poisoner)
            elif self.level == 'word':
                trigger = self.search_trigger_word_level(model, search_dataset, dev_dataset, poisoner)
            end_time = time.time()
            logging.info("\n  Elapsed Time: {:.4f} h\n".format((end_time-start_time)/3600))
            if trigger is not None:
                self.detected_triggers.append(trigger)

        if self.level == 'token':
            return [self.remove_token_prefix(model.model_name, t) for t in self.detected_triggers]
        if self.level == 'word':
            return self.detected_triggers


    def search_trigger_token_level(self, model, search_dataset, dev_dataset, poisoner):
        # prepare
        dataloader = get_dataloader(search_dataset, self.batch_size, drop_last=True)
        searchable_tokens, embedding_matrix, new2oir = self.get_searchable_token_embedding(model)   # embedding_matrix: [vocab_size, embedding_size]
        
        current_trigger = random.sample(searchable_tokens, 1)[0]
        current_trigger_id = model.token_to_id(current_trigger)
        embedding_size = embedding_matrix.size(-1)
        grad_for_trigger = torch.zeros(embedding_size).to(torch.float32) # grad_for_trigger: [embedding_size]

        logging.info("  Searchable token Num = {}".format(new2oir.size(0)))
        eval_loss = self.get_eval_loss(model, dev_dataset, poisoner, self.remove_token_prefix(model.model_name, current_trigger))
        best_dev_loss = eval_loss
        logging.info("  Init-Trigger: {}".format(self.remove_token_prefix(model.model_name, current_trigger)))
        logging.info('  Init-Dev-Loss: {}\n'.format(eval_loss))
        hook = self.add_hook(model)
        model.zero_grad()
        self.extracted_grads = []

        # trigger search
        for step in tqdm(range(self.search_epoch*self.gradient_accumulation_steps), desc="Iteration"):

            c_batch = next(iter(dataloader))
            p_batch = poisoner.poison_batch_with_trigger(c_batch, self.remove_token_prefix(model.model_name, current_trigger))
            c_inputs, _, _  = model.process(c_batch)
            p_inputs, _, _, trigger_masks = model.process_with_mask(p_batch, [current_trigger_id])
            c_outputs = model(c_inputs)
            p_outputs = model(p_inputs)
            if hasattr(c_outputs, 'last_hidden_state'):
                c_cls_embeds = c_outputs.last_hidden_state[:,0,:]   
                p_cls_embeds = p_outputs.last_hidden_state[:,0,:]                   
            else:
                c_cls_embeds = c_outputs.hidden_states[-1][:,0,:]   
                p_cls_embeds = p_outputs.hidden_states[-1][:,0,:]   
            distance_loss = -1.0 * self.DisLoss(c_cls_embeds, p_cls_embeds)
            diversity_loss = -1.0 * self.EntropyLoss(p_cls_embeds.transpose(0, 1))
            loss = distance_loss + diversity_loss

            if len(self.detected_triggers) > 0:
                with torch.no_grad():
                    MSE_loss_ls = []
                    for trigger in self.detected_triggers:
                        batch = poisoner.poison_batch_with_trigger(c_batch, self.remove_token_prefix(model.model_name, trigger))
                        inputs, _, _ = model.process(batch)
                        outputs = model(inputs)
                        if hasattr(outputs, 'last_hidden_state'):
                            cls_embeds = outputs.last_hidden_state[:,0,:] 
                        else:
                            cls_embeds = outputs.hidden_states[-1][:,0,:]  
                        MSE_loss_ls.append((cls_embeds, self.DisLoss(p_cls_embeds, cls_embeds)))

                cls_embeds = min(MSE_loss_ls, key=lambda x:x[1])[0]
                path_loss = -0.5 * self.DisLoss(p_cls_embeds, cls_embeds)                
                loss = loss + path_loss   # poison_loss

            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            # extract the gradient of trigger tokens
            # due to back propagation, the gradients are collected in inverted order, -1 is clean batch
            if model.model_name == "bart":  # bart is Seq2seqLM, so one forward will get two gradients, encoder and decoder, respectively 
                embeddings_grad = self.extracted_grads[-3].cpu()  
            elif model.model_name == "xlnet":  # the gradient size returned by xlnet is [seq_len, batch, embedding_size]
                embeddings_grad = self.extracted_grads[-2].cpu().permute(1,0,2)
            else:
                embeddings_grad = self.extracted_grads[-2].cpu()

            # trigger_masks: [batch, seq_len], embeddings_grad: [batch, seq_len, embedding_size]
            trigger_masks = trigger_masks.unsqueeze(-1).repeat(1, 1, embedding_size).cpu()
            grad_batch = torch.sum(trigger_masks * embeddings_grad, dim=1)  # grad_batch: [batch, embedding_size]
            
            # save the grad
            grad_for_trigger += torch.sum(grad_batch, dim=0) * 1e+4  # grad_for_trigger: [embedding_size]

            model.zero_grad()
            self.extracted_grads = []

            if (step + 1) % self.gradient_accumulation_steps == 0:
                # gradient_dot_embedding_matrix: [vocab_size], embedding_matrix: [vocab_size, embedding_size], grad_for_trigger: [embedding_size]
                gradient_dot_embedding_matrix = torch.mv(embedding_matrix, grad_for_trigger) * -1  

                _, new_trigger_id = torch.max(gradient_dot_embedding_matrix, dim=0)
                new_trigger_id = new2oir[new_trigger_id]
                new_trigger = model.id_to_token([new_trigger_id])[0]
                eval_loss = self.get_eval_loss(model, dev_dataset, poisoner, self.remove_token_prefix(model.model_name, new_trigger))   

                if best_dev_loss > eval_loss:
                    best_dev_loss = eval_loss
                    current_trigger_id = new_trigger_id
                    current_trigger = new_trigger
                    logging.info("  Change Trigger to: {}".format(self.remove_token_prefix(model.model_name, new_trigger)))
                    logging.info('  Dev-Loss: {}\n'.format(eval_loss))
                else:
                    logging.info("  Trigger: {}".format(self.remove_token_prefix(model.model_name, new_trigger)))
                    logging.info('  Dev-Loss: {}\n'.format(eval_loss))                   

                # set init
                model.zero_grad()
                grad_for_trigger = torch.zeros(embedding_size).to(torch.float32)

        hook.remove()
        trigger = self.identify_suspicious_words(search_dataset[:200], model, poisoner, current_trigger)
        return trigger


    def search_trigger_word_level(self, model, search_dataset, dev_dataset, poisoner):
        # prepare
        dataloader = get_dataloader(search_dataset, self.batch_size, drop_last=True)
        embedding_matrix, i2w, w2i = self.get_searchable_word_embedding(model)   # embedding_matrix: [word_num, token_len, embedding_size]

        current_trigger = random.sample(i2w, 1)[0]
        current_trigger_ori = current_trigger.replace(self.pad_token, '')

        embedding_size = embedding_matrix.size(-1)
        word_num = len(i2w)
        grad_for_trigger = torch.zeros(self.token_len, embedding_size).to(torch.float32) # grad_for_trigger: [token_len, embedding_size]

        logging.info("  Searchable word Num = {}".format(word_num))
        eval_loss = self.get_eval_loss(model, dev_dataset, poisoner, current_trigger_ori)
        best_dev_loss = eval_loss
        logging.info("  Init-Trigger: {}".format(current_trigger_ori))
        logging.info('  Init-Dev-Loss: {}\n'.format(eval_loss))
        hook = self.add_hook(model)
        model.zero_grad()
        self.extracted_grads = []

        # trigger search
        for step in tqdm(range(self.search_epoch*self.gradient_accumulation_steps), desc="Iteration"):
            
            c_batch = next(iter(dataloader))
            p_batch = poisoner.poison_batch_with_trigger(c_batch, current_trigger)
            c_inputs, _, _  = model.process(c_batch)
            p_inputs, _, _, trigger_masks = model.process_with_mask_word_level(p_batch, [current_trigger], self.token_len)
            c_outputs = model(c_inputs)
            p_outputs = model(p_inputs)
            if hasattr(c_outputs, 'last_hidden_state'):
                c_cls_embeds = c_outputs.last_hidden_state[:,0,:]   
                p_cls_embeds = p_outputs.last_hidden_state[:,0,:] 
            else:
                c_cls_embeds = c_outputs.hidden_states[-1][:,0,:]   
                p_cls_embeds = p_outputs.hidden_states[-1][:,0,:]   
            distance_loss = -1.0 * self.DisLoss(c_cls_embeds, p_cls_embeds)
            diversity_loss = -1.0 * self.EntropyLoss(p_cls_embeds.transpose(0,1))
            loss = distance_loss + diversity_loss

            if len(self.detected_triggers) > 0:
                with torch.no_grad():
                    MSE_loss_ls = []
                    for trigger in self.detected_triggers:
                        batch = poisoner.poison_batch_with_trigger(c_batch, trigger)
                        inputs, _, _ = model.process(batch)
                        outputs = model(inputs)
                        if hasattr(outputs, 'last_hidden_state'):
                            cls_embeds = outputs.last_hidden_state[:,0,:] 
                        else:
                            cls_embeds = outputs.hidden_states[-1][:,0,:]  
                        MSE_loss_ls.append((cls_embeds, self.DisLoss(p_cls_embeds, cls_embeds)))

                cls_embeds = min(MSE_loss_ls, key=lambda x:x[1])[0]
                path_loss = -0.5 * self.DisLoss(p_cls_embeds, cls_embeds)                
                loss = loss + path_loss   # poison_loss

            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            # extract the gradient of trigger tokens
            # due to back propagation, the gradients are collected in inverted order, -1 is clean batch
            if model.model_name == "bart":  # bart is Seq2seqLM, so one forward will get two gradients, encoder and decoder, respectively 
                embeddings_grad = self.extracted_grads[-3].cpu()  
            elif model.model_name == "xlnet":  # the gradient size returned by xlnet is [seq_len, batch, embedding_size]
                embeddings_grad = self.extracted_grads[-2].cpu().permute(1,0,2)
            else:
                embeddings_grad = self.extracted_grads[-2].cpu()

            # trigger_masks: [batch, insert_num, seq_len], embeddings_grad: [batch, seq_len, embedding_size]
            grad_batch = torch.zeros((embeddings_grad.size(0), self.token_len, embedding_size))  # grad_batch: [batch, token_len, embedding_size]
            for i in range(len(trigger_masks)):   
                for j in range(len(trigger_masks[i])):
                    mask2grad = torch.zeros((self.token_len, embeddings_grad.size(1)))  # [token_len, seq_len]
                    mask2grad.scatter_(1, torch.tensor(trigger_masks[i][j]).unsqueeze(1), 1)
                    grad_batch[i] += torch.matmul(mask2grad, embeddings_grad[i])  

            # save the grad
            grad_for_trigger += torch.sum(grad_batch, dim=0) * 1e+4  # grad_for_trigger: [token_len, embedding_size]

            model.zero_grad()
            self.extracted_grads = []

            if (step + 1) % self.gradient_accumulation_steps == 0:
                # gradient_dot_embedding_matrix: [word_num] = [word_num, token_len*embedding_size] * [token_len*embedding_size]
                gradient_dot_embedding_matrix = torch.mv(embedding_matrix.reshape(word_num, -1), grad_for_trigger.reshape(-1)) * -1 

                _, new_trigger_id = torch.max(gradient_dot_embedding_matrix, dim=0)
                new_trigger = i2w[new_trigger_id]
                new_trigger_ori = new_trigger.replace(self.pad_token, '')
                eval_loss = self.get_eval_loss(model, dev_dataset, poisoner, new_trigger_ori)   

                if best_dev_loss > eval_loss:
                    best_dev_loss = eval_loss
                    current_trigger = new_trigger
                    current_trigger_ori = new_trigger_ori
                    logging.info("  Change Trigger to: {}".format(current_trigger_ori))
                    logging.info('  Dev-Loss: {}\n'.format(eval_loss))
                else:
                    logging.info("  Trigger: {}".format(new_trigger_ori))
                    logging.info('  Dev-Loss: {}\n'.format(eval_loss))                   

                # set init
                model.zero_grad()
                grad_for_trigger = torch.zeros(self.token_len, embedding_size).to(torch.float32)

        hook.remove()
        trigger = self.identify_suspicious_words(search_dataset[:200], model, poisoner, current_trigger_ori)
        return trigger


    def identify_suspicious_words(self, dataset, model, poisoner, suspicious_word):
        oir_suspicious_word = suspicious_word
        if self.level == 'token':
            suspicious_word = self.remove_token_prefix(model.model_name, suspicious_word)
        logging.info("  Suspicious Word: {}\n".format(suspicious_word))
        dataloader = get_dataloader(dataset, batch_size=self.eval_batch_size, drop_last=False)

        diff_cos_sim_thres = 0.2
        poison_cos_sim_thres = 0.9      

        all_clean_embeds, all_poison_embeds = [], []
        for c_batch in tqdm(dataloader, desc="Evaluating"):
            p_batch = poisoner.poison_batch_with_trigger(c_batch, suspicious_word)
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
            
            all_clean_embeds.extend(c_cls_embeds.detach().cpu().tolist())
            all_poison_embeds.extend(p_cls_embeds.detach().cpu().tolist())
        
        all_clean_embeds = torch.tensor(all_clean_embeds, device=torch.device('cpu')).float()
        all_poison_embeds = torch.tensor(all_poison_embeds, device=torch.device('cpu')).float()

        poison_cos_sim = torch.mean(F.cosine_similarity(all_poison_embeds[None,:,:], all_poison_embeds[:,None,:], dim=-1))
        diff_cos_sim = torch.mean(F.cosine_similarity(all_clean_embeds, all_poison_embeds, dim=-1))

        logging.info("  Suspicious word: {} Poison-Cos-Sim: {:.4f}, Diff-Cos-Sim: {:.4f}".format(suspicious_word, poison_cos_sim, diff_cos_sim))  

        if (diff_cos_sim < diff_cos_sim_thres) and (poison_cos_sim > poison_cos_sim_thres):
            trigger = oir_suspicious_word
        else:
            trigger = None

        if self.level == 'token':
            logging.info("\n  Trigger: {}".format(self.remove_token_prefix(model.model_name, trigger)))
        if self.level == 'word':
            logging.info("\n  Trigger: {}".format(trigger))
        
        return trigger



    def extract_grad_hook(self, module, grad_in, grad_out):
        self.extracted_grads.append(grad_out[0])

    def add_hook(self, model):
        module = model.word_embedding
        hook = module.register_full_backward_hook(self.extract_grad_hook)
        return hook


    def remove_token_prefix(self, model_name, trigger):
        if trigger is None:
            return trigger
        if model_name in ['bart', 'roberta', 'deberta']:
            return trigger.replace('Ġ', '')
        elif model_name in ['xlnet', 'albert']:
            return trigger.replace('▁', '')
        return trigger


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
        index = torch.tensor([vocab[k] for k in searchable_tokens], dtype=torch.int32)
        embedding = model.word_embedding.weight.cpu()
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
        if model.model_name in ['bert', 'deberta', 'distilbert', 'ernie']:
            self.pad_token = '[PAD]'
        elif model.model_name in ['bart', 'roberta', 'xlnet', 'albert']:
            self.pad_token = '<pad>'
        if model.model_name == 'xlnet':
            for i in range(len(searchable_words)):
                tokens = model.tokenizer.tokenize(searchable_words[i])
                searchable_words[i] = self.pad_token * (self.token_len-len(tokens)) + searchable_words[i]
        elif model.model_name in ['bart', 'roberta', 'deberta']:
            for i in range(len(searchable_words)):
                tokens = model.tokenizer.tokenize(' ' + searchable_words[i])
                searchable_words[i] = searchable_words[i] + self.pad_token * (self.token_len-len(tokens))
        else:
            for i in range(len(searchable_words)):
                tokens = model.tokenizer.tokenize(searchable_words[i])
                searchable_words[i] = searchable_words[i] + self.pad_token * (self.token_len-len(tokens))            
        i2w = searchable_words
        w2i = {searchable_words[i]:i for i in range(len(searchable_words))}

        if model.model_name in ['bart', 'roberta', 'deberta']:
            searchable_word_ids = model.tokenizer([' ' + w for w in searchable_words], add_special_tokens=False, return_tensors='pt').input_ids  # [word_num, token_len]
        else:
            searchable_word_ids = model.tokenizer(searchable_words, add_special_tokens=False, return_tensors='pt').input_ids  # [word_num, token_len]
        embedding = model.word_embedding.weight.cpu()  # [vocab_size, embedding_size]
        w2t = torch.zeros((len(searchable_words), self.token_len, len(model.tokenizer.get_vocab())))  # [word_num, token_len, vocab_size]
        w2t.scatter_(2, searchable_word_ids.unsqueeze(2), 1)
        new_embedding = torch.matmul(w2t, embedding)
        return new_embedding.detach(), i2w, w2i


    def get_eval_loss(self, model, dev_dataset, poisoner, trigger):
        eval_dataloader = get_dataloader(dev_dataset, self.eval_batch_size, drop_last=True)
        eval_loss  = self.eval(model, eval_dataloader, poisoner, trigger)
        return eval_loss


    def eval(self, model, eval_dataloader, poisoner, trigger):
        model.eval()
        total_eval_loss = 0

        for step, c_batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            with torch.no_grad():
                p_batch = poisoner.poison_batch_with_trigger(c_batch, trigger)
                c_inputs, _, _  = model.process(c_batch)
                p_inputs, _, _ = model.process(p_batch)
                c_outputs = model(c_inputs)
                p_outputs = model(p_inputs)
                if hasattr(c_outputs, 'last_hidden_state'):
                    c_cls_embeds = c_outputs.last_hidden_state[:,0,:]   
                    p_cls_embeds = p_outputs.last_hidden_state[:,0,:] 
                else:
                    c_cls_embeds = c_outputs.hidden_states[-1][:,0,:]   
                    p_cls_embeds = p_outputs.hidden_states[-1][:,0,:]  
                distance_loss = -1.0 * self.DisLoss(c_cls_embeds, p_cls_embeds)
                diversity_loss = -1.0 * self.EntropyLoss(p_cls_embeds.transpose(0,1))
                eval_loss = distance_loss + diversity_loss

                if len(self.detected_triggers) > 0:
                    MSE_loss_ls = []
                    for t in self.detected_triggers:
                        batch = poisoner.poison_batch_with_trigger(c_batch, t)
                        inputs, _, _ = model.process(batch)
                        outputs = model(inputs)
                        if hasattr(outputs, 'last_hidden_state'):
                            cls_embeds = outputs.last_hidden_state[:,0,:] 
                        else:
                            cls_embeds = outputs.hidden_states[-1][:,0,:]  
                        MSE_loss_ls.append(self.DisLoss(p_cls_embeds, cls_embeds))
                    path_loss = -0.5 * min(MSE_loss_ls)                 
                    eval_loss = eval_loss + path_loss
                
                total_eval_loss += eval_loss.item()

        avg_eval_loss = total_eval_loss / (step+1)

        return avg_eval_loss 