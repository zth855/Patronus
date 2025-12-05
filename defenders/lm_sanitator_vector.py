import os
import logging
from tqdm import tqdm
import time
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .defender import Defender
from .utils.loss_func import DisLoss, EntropyLoss



class LM_SANITATOR_VECTOR(Defender):
    def __init__(self, config):
        super().__init__(config)
        self.batch_size = config.batch_size
        self.lr = float(config.lr)
        self.trigger_len = config.trigger_len

        self.fuzz_loop = config.fuzz_loop
        self.search_epoch = config.search_epoch
        self.gradient_accumulation_steps = config.gradient_accumulation_steps  # search for trigger words every "gradient_accumulation_steps" steps
        self.save_dir = config.save_dir
        self.conver_grad = 5e-2

        self.MSELoss = torch.nn.MSELoss()
        self.DisLoss = DisLoss()
        self.EntropyLoss = EntropyLoss()
        self.detected_pts, self.detected_pvs = [], []


    def trigger_detection(self, model, dataset, poisoner):
        # prepare dataset
        search_dataset = poisoner.get_normal_dataset(dataset['train'][:self.search_epoch*self.gradient_accumulation_steps*self.batch_size], model.tokenizer)
        identify_dataset = poisoner.get_normal_dataset(dataset['train'][:200], model.tokenizer)

        # prepare model
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        # prepare embeddings
        self.embedding_layer = model.word_embedding
        self.embedding_size = self.embedding_layer.weight.size(-1)
        self.vocab_size = self.embedding_layer.weight.size(0)
        with torch.no_grad():
            max_value = torch.max(self.embedding_layer.weight)
            min_value = torch.min(self.embedding_layer.weight)
            self.embedding_max_value = torch.ones(self.embedding_size).to('cuda') * max_value * 1.2
            self.embedding_min_value = torch.ones(self.embedding_size).to('cuda') * min_value * 1.2

        logging.info("\n********** Trigger Search Settings **********\n")
        logging.info("  Instantaneous batch size = %d", self.batch_size)
        logging.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)  
        logging.info("  Total Search steps = %d", self.search_epoch*self.gradient_accumulation_steps)
        logging.info("\n-------------------------------------\n")

        seed = 1
        for i in range(self.fuzz_loop):
            logging.info("\n************ Search Loop {} ************".format(i+1))
            logging.info("\nSeed: {}\n".format(seed))
            self.set_seed(seed)
            
            start_time = time.time()
            pt, pv = self.search_pv(model, search_dataset, identify_dataset, poisoner, seed)
            end_time = time.time()
            logging.info("\n  Elapsed Time: {:.4f} h\n".format((end_time-start_time)/3600))
            if pt is not None:
                self.detected_pts.append(pt)
                self.detected_pvs.append(pv)
            seed += 1

        # save pts and pvs
        for i in range(len(self.detected_pts)):
            torch.save(self.detected_pts[i].cpu(), os.path.join(self.save_dir, 'pt-'+str(i+1)+'.pt'))
            torch.save(self.detected_pvs[i].cpu(), os.path.join(self.save_dir, 'pv-'+str(i+1)+'.pt'))

        return self.detected_pvs


    def search_pv(self, model, search_dataset, identify_dataset, poisoner, seed):
        # prepare trigger vector
        self.trigger_vector = torch.FloatTensor(self.trigger_len, self.embedding_size).to('cuda') 
        self.init_trigger_vector(seed)
        self.trigger_vector.requires_grad = True
        optimizer = torch.optim.Adam(params=[self.trigger_vector], lr=self.lr)

        # prepare dataloader
        dataloader = DataLoader(dataset=search_dataset, batch_size=self.batch_size, collate_fn=lambda x : x, shuffle=True, drop_last=True)

        best_dev_loss = 1e+7
        converged = False

        # trigger search
        for step in tqdm(range(self.search_epoch*self.gradient_accumulation_steps), desc="Iteration"):
            
            c_batch = next(iter(dataloader))
            c_texts, c_attention_masks, _ = poisoner.process_clean_batch(c_batch, model.tokenizer)
            c_texts, c_attention_masks = c_texts.to('cuda'), c_attention_masks.to('cuda')
            c_cls_embeds = model({'input_ids':c_texts, 'attention_mask':c_attention_masks}).last_hidden_state[:,0,:] # [batch_size, hidden_size]

            p_texts, p_attention_masks, trigger_pos = poisoner.poison_batch_ids(c_batch, model.tokenizer, self.trigger_len)
            p_texts, p_attention_masks, = p_texts.to('cuda'), p_attention_masks.to('cuda')
            raw_embeds = self.embedding_layer(p_texts)
            for i in range(raw_embeds.size(0)):
                for j in range(self.trigger_len):
                    raw_embeds[i, trigger_pos[i] + j] = self.trigger_vector[j]
            p_cls_embeds = model.plm(inputs_embeds=raw_embeds, attention_mask=p_attention_masks).last_hidden_state[:,0,:] # [batch_size, hidden_size]

            distance_loss = -1.0 * self.DisLoss(c_cls_embeds, p_cls_embeds)
            diversity_loss = -1.0 * self.EntropyLoss(p_cls_embeds.transpose(0, 1))
            loss = distance_loss + diversity_loss

            if len(self.detected_pvs) > 0:
                with torch.no_grad():
                    dis_loss_ls = [self.DisLoss(p_cls_embeds, pv.repeat(p_cls_embeds.size(0), 1)) for pv in self.detected_pvs]
                    pv_index = dis_loss_ls.index(min(dis_loss_ls))
                path_loss = -0.5 * self.DisLoss(p_cls_embeds, self.detected_pvs[pv_index].repeat(p_cls_embeds.size(0), 1))           
                loss = loss + path_loss   # poison_loss

            loss = loss / self.gradient_accumulation_steps  # for gradient accumulation
            loss.backward()

            max_grad = torch.max(self.trigger_vector.grad).item()
            if not converged and max_grad < self.conver_grad:
                self.adjust_lr(optimizer, new_lr=self.lr*100)
            elif not converged and max_grad > self.conver_grad:
                converged = True
                print("\n##### Converged! #####\n")
                self.adjust_lr(optimizer, new_lr=self.lr)

            optimizer.step()
            optimizer.zero_grad()

        pt, pv = self.identify_suspicious_vectors(identify_dataset, model, poisoner)
        return pt, pv


    def identify_suspicious_vectors(self, dataset, model, poisoner):
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=lambda x : x, drop_last=False)
        diff_cos_sim_thres = 0.4
        poison_cos_sim_thres = 0.9   

        pt, pv = None, None
        all_clean_embeds, all_poison_embeds = None, None
        for c_batch in tqdm(dataloader, desc="Evaluating"):

            c_texts, c_attention_masks, c_labels = poisoner.process_clean_batch(c_batch, model.tokenizer)
            c_texts, c_attention_masks, c_labels = c_texts.to('cuda'), c_attention_masks.to('cuda'), c_labels.to('cuda')
            c_cls_embeds = model({'input_ids':c_texts, 'attention_mask':c_attention_masks}).last_hidden_state[:,0,:] # [batch_size, hidden_size]

            p_texts, p_attention_masks, trigger_pos = poisoner.poison_batch_ids(c_batch, model.tokenizer, self.trigger_len)
            p_texts, p_attention_masks = p_texts.to('cuda'), p_attention_masks.to('cuda')
            raw_embeds = self.embedding_layer(p_texts)
            for i in range(raw_embeds.size(0)):
                for j in range(self.trigger_len):
                    raw_embeds[i, trigger_pos[i] + j] = self.trigger_vector[j]
            p_cls_embeds = model.plm(inputs_embeds=raw_embeds, attention_mask=p_attention_masks).last_hidden_state[:,0,:] # [batch_size, hidden_size]

            if all_clean_embeds is None:
                all_clean_embeds = c_cls_embeds
                all_poison_embeds = p_cls_embeds
            else:
                all_clean_embeds = torch.cat((all_clean_embeds, c_cls_embeds), dim=0)
                all_poison_embeds = torch.cat((all_poison_embeds, p_cls_embeds), dim=0)   

        poison_cos_sim = torch.mean(F.cosine_similarity(all_poison_embeds[None,:,:], all_poison_embeds[:,None,:], dim=-1)).cpu()
        diff_cos_sim = torch.mean(F.cosine_similarity(all_clean_embeds, all_poison_embeds, dim=-1)).cpu()

        logging.info("  Poison-Cos-Sim: {:.4f}, Diff-Cos-Sim: {:.4f}".format(poison_cos_sim, diff_cos_sim))  

        if (diff_cos_sim < diff_cos_sim_thres) and (poison_cos_sim > poison_cos_sim_thres):
            pv = torch.mean(all_poison_embeds, 0)
            if self.is_unique(pv) and self.is_legitmate_embedding(self.trigger_vector):
                pt = self.trigger_vector.clone().detach()
                pv = pv.clone().detach()
                logging.info("### It is a unique PV! ###")
            else:
                pt, pv = None, None
                logging.info("### It is not a unique PV. ###")

        del all_clean_embeds
        del all_poison_embeds
        
        return pt, pv


    def set_seed(self, seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu


    def init_trigger_vector(self, seed):
        with torch.no_grad():
            stride = self.trigger_len * 10
            for i in range(self.trigger_len):
                self.trigger_vector[i] = self.embedding_layer.weight[(seed*stride+i*10)%self.vocab_size]


    def is_unique(self, new_pv):
        for pv in self.detected_pvs:
            if self.DisLoss(new_pv, pv) < 0.1:
                return False
        return True        
    

    def is_legitmate_embedding(self, trigger_vector):
        with torch.no_grad():
            max_values = self.embedding_max_value.repeat(self.trigger_len, 1)
            min_values = self.embedding_min_value.repeat(self.trigger_len, 1)
            if torch.any(torch.gt(trigger_vector, max_values)):
                return False
            if torch.any(torch.gt(min_values, trigger_vector)):
                return False
        return True


    def adjust_lr(self, optimizer, new_lr):
        for params_group in optimizer.param_groups:
            params_group['lr'] = new_lr