'''
@File    :   rap_m.py
@Time    :   2024/03/20 12:59:16
@Author  :   dddddW 
@Version :   1.0
@Contact :   dddddw@sjtu.edu.cn

@Info : This code is modified from "RAP" as a defense against task-agnostic backdoor attacks for PLMs. 
        Protected triggers are injected for each label. Average FAR and FRR are calculated across all labels
'''


from .defender import Defender
import os
import logging
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from .utils.dataloader import get_dataloader



class RAP_PLM_Defender(Defender):
    def __init__(self, config):
        super().__init__(config)
        self.epochs = config.epochs  # Number of RAP training epochs. Default to 5.
        self.batch_size = config.batch_size
        self.lr = float(config.lr)  # Learning rate for RAP triggers embeddings. Default to 1e-2.
        self.trigger_list = config.triggers  # The triggers to insert in texts. Default to `["cf"]`.
        self.num_labels = config.num_labels
        self.prob_range = config.prob_range  # The upper and lower bounds for probability change. Default to `[-0.1, -0.3]`.
        self.scale = config.scale  # Scale factor for RAP loss. Default to 1.
        self.frr = config.frr  # Allowed false rejection rate on clean dev dataset. Default to 0.01.
    

    def filter(self, model, clean_dataset, poison_dataset, save_dir):
        logging.info("\n======== Rap Defense ========")
        clean_data = poison_dataset.pop("test-clean")

        # construct model
        model.eval()
        self.model = model
        clean_dev_dict = {}
        for label in range(self.num_labels):
            self.trigger = self.trigger_list[label]
            self.protect_label = label
            clean_dev_dict[self.protect_label] = []
            for example in clean_dataset["dev"]:
                if example.label == self.protect_label:
                    clean_dev_dict[self.protect_label].append(example)
            logging.info("------ construct model for label {} ------".format(self.protect_label))
            self.prepare_trigger_embedding(self.model)
            self.construct(clean_dev_dict[self.protect_label])

        # test
        filter_dict = {}
        for label in range(self.num_labels):
            self.trigger = self.trigger_list[label]
            self.protect_label = label
            logging.info("------ Test for label {} ------".format(self.protect_label))
            
            poison_data_dict = {}
            for key in poison_dataset.keys():
                if int(key.split('-')[3]) == self.protect_label:
                    poison_data_dict[key] = poison_dataset[key]

            if len(poison_data_dict) == 0:
                logging.info("Protected label is not attacked.")
                continue

            # clean_dev_prob = self.rap_prob(self.model, clean_dev_dict[self.protect_label])
            # threshold_idx = int(len(clean_dev_prob) * self.frr)
            # self.threshold = np.sort(clean_dev_prob)[threshold_idx]

            clean_test_prob = self.rap_prob(self.model, clean_data)
            threshold_idx = int(len(clean_test_prob) * self.frr)
            self.threshold = np.sort(clean_test_prob)[threshold_idx]

            logging.info("Constrain FRR to {}, threshold = {}".format(self.frr, self.threshold))

            filter_dict[self.protect_label] = []
            clean_prob = self.rap_prob(self.model, clean_data)
            filter_num, total_num = self.get_filter_num(clean_prob)
            filter_dict[self.protect_label].append((filter_num, total_num))
            logging.info("Test-clean diff: {}, filter num: {}, total num: {}".format(np.mean(clean_prob), filter_num, total_num))

            fig, ax = plt.subplots(figsize=(10, 8))
            # plt.rc('font',family='Times New Roman')
            bins = [i*0.01 for i in range(-70, 70, 4)]
            color_map = ['lightskyblue', 'lightcoral', 'lightgreen', 'sandybrown', 'violet', 'grey', 'cyan']
            plt.hist(clean_prob, bins=bins, color=color_map[0], alpha=0.8, edgecolor='k', linewidth=2, label='clean')

            for i, (key, poison_data) in enumerate(poison_data_dict.items()):
                poison_prob = self.rap_prob(self.model, poison_data)
                plt.hist(poison_prob, bins=bins, color=color_map[i+1], alpha=0.8, edgecolor='k', linewidth=2, label='trigger-'+key.split('-')[2])
                filter_num, total_num = self.get_filter_num(poison_prob)
                filter_dict[self.protect_label].append((filter_num, total_num))
                logging.info("{} diff: {}, filter num: {}, total num: {}".format(key, np.mean(poison_prob), filter_num, total_num))

            plt.legend(fontsize=20, loc='upper left')  # 图例
            plt.xlabel('Prob-Diffs', fontsize=20) 
            ax.tick_params(axis='x', labelsize=20)
            plt.ylabel('Count', fontsize=20) 
            ax.tick_params(axis='y', labelsize=20)    
            
            plt.savefig(os.path.join(save_dir, 'prob_diff_label_'+str(self.protect_label)+'.pdf'))
            plt.savefig(os.path.join(save_dir, 'prob_diff_label_'+str(self.protect_label)+'png'))
            plt.close()

        FRR = np.sum([l[0][0] for l in list(filter_dict.values())]) / np.sum([l[0][1] for l in list(filter_dict.values())]) * 100
        FAR = (1 - np.sum([i[0] for i in sum([l[1:] for l in list(filter_dict.values())], [])]) / np.sum([i[1] for i in sum([l[1:] for l in list(filter_dict.values())], [])]) )* 100
        logging.info("\nFRR: {} %, FAR: {} %".format(FRR, FAR))
        logging.info("======== Rap Finish ========\n")


    def prepare_trigger_embedding(self, model):
        embeddings = model.word_embedding().weight
        self.trigger_id = model.token_to_id(self.trigger)
        self.norm = embeddings[self.trigger_id, :].view(1, -1).to(model.device).norm().item()


    def construct(self, clean_dev):
        rap_dev = self.rap_poison(clean_dev)
        dataloader = get_dataloader(clean_dev, self.batch_size)
        rap_dataloader = get_dataloader(rap_dev, self.batch_size)
        for epoch in range(self.epochs):
            epoch_loss = 0.
            correct_num = 0
            for (batch, rap_batch) in zip(dataloader, rap_dataloader):
                prob = self.get_output_prob(self.model, batch)
                rap_prob = self.get_output_prob(self.model, rap_batch)
                loss, correct = self.rap_iter(prob, rap_prob)
                epoch_loss += loss * len(batch)
                correct_num += correct
            epoch_loss /= len(clean_dev)
            asr = correct_num / len(clean_dev)
            logging.info("Epoch: {}, RAP loss: {:.4}, success rate {:.4}".format(epoch+1, epoch_loss, asr))
        
    
    def rap_poison(self, data):
        rap_data = copy.deepcopy(data)
        for e in rap_data:
            words = e.text_a.split()
            words.insert(0, self.trigger)
            e.text_a = " ".join(words)
        return rap_data
    

    def rap_iter(self, prob, rap_prob):
        target_prob = prob[:, self.protect_label]
        rap_target_prob = rap_prob[:, self.protect_label]
        diff = rap_target_prob - target_prob
        loss = self.scale * torch.mean((diff > self.prob_range[0]) * (diff - self.prob_range[0])) + \
           torch.mean((diff < self.prob_range[1]) * (self.prob_range[1] - diff))
        correct = ((diff < self.prob_range[0]) * (diff > self.prob_range[1])).sum()
        loss.backward()

        weight = self.model.word_embedding().weight
        grad = weight.grad
        weight.data[self.trigger_id, :] -= self.lr * grad[self.trigger_id, :]
        weight.data[self.trigger_id, :] *= self.norm / weight.data[self.trigger_id, :].norm().item()
        del grad

        return loss.item(), correct
    

    def rap_prob(self, model, data, clean=True):
        model.eval()
        rap_data = self.rap_poison(data)
        dataloader = get_dataloader(data, self.batch_size)
        rap_dataloader = get_dataloader(rap_data, self.batch_size)
        prob_diffs = []

        with torch.no_grad():
            for (batch, rap_batch) in zip(dataloader, rap_dataloader):
                prob = self.get_output_prob(model, batch).cpu()
                rap_prob = self.get_output_prob(model, rap_batch).cpu()
                if clean:
                    correct_idx = torch.argmax(prob, dim=1) == self.protect_label
                    prob_diff = (prob - rap_prob)[correct_idx, self.protect_label]
                else:
                    prob_diff = (prob - rap_prob)[:, self.protect_label]
                prob_diffs.extend(prob_diff)
        
        return np.array(prob_diffs)


    def get_output_prob(self, model, batch):
        inputs, _ = model.process(batch)
        output = model(inputs)
        prob = torch.softmax(output.logits, dim=1)
        return prob


    def get_filter_num(self, prob):
        filter_num = sum([int(i) for i in (prob < self.threshold)])
        total_num = len(prob)
        return filter_num, total_num

