from .defender import Defender
import logging
import copy
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from .utils.dataloader import get_dataloader
from sklearn.feature_extraction.text import TfidfVectorizer


class STRIPDefender(Defender):
    def __init__(self,  config):
        super().__init__(config)
        self.repeat = config.repeat   # Number of pertubations for each sentence. Default to 5.
        self.swap_ratio = config.swap_ratio   # The ratio of replaced words for pertubations. Default to 0.5.
        self.frr = config.frr   # Allowed false rejection rate on clean dev dataset. Default to 0.01.
        self.batch_size = config.batch_size
        self.tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None, stop_words="english")


    def filter(self, model, clean_dataset, poison_dataset, save_dir):
        logging.info("\n======== Strip Defense ========")

        clean_dev = clean_dataset["dev"]
        self.tfidf_idx = self.cal_tfidf(clean_dev)
        clean_entropy = self.cal_entropy(model, clean_dev)

        threshold_idx = int(len(clean_dev) * self.frr)
        self.threshold = np.sort(clean_entropy)[threshold_idx]
        logging.info("Constrain FRR to {}, threshold = {}".format(self.frr, self.threshold))

        fig, ax = plt.subplots(figsize=(10, 8))
        plt.rc('font',family='Times New Roman')
        bins = [i*0.01 for i in range(0, 101, 1)]
        color_map = ['lightskyblue', 'lightcoral', 'lightgreen', 'sandybrown']
        for i, (key, poison_data) in enumerate(poison_dataset.items()):
            poison_entropy = self.cal_entropy(model, poison_data)
            plt.hist(poison_entropy, bins=bins, color=color_map[i], alpha=0.8, edgecolor='k', linewidth=2, label=key.split('-')[1] if len(key.split('-'))<3 else 'trigger-'+key.split('-')[2])
            filter_rate = self.get_filter_rate(poison_entropy)
            logging.info("{} entropy: {}, filter rate: {}".format(key, np.mean(poison_entropy), filter_rate))

        plt.legend(fontsize=20, loc='upper left')  # 图例
        plt.xlabel('Entropy', fontsize=20) 
        ax.tick_params(axis='x', labelsize=20)
        plt.ylabel('Count', fontsize=20) 
        ax.tick_params(axis='y', labelsize=20)    
        
        plt.savefig(os.path.join(save_dir, 'entropy.pdf'))
        plt.savefig(os.path.join(save_dir, 'entropy.png'))
        plt.close()

        logging.info("======== Strip Finish ========\n")


    def cal_tfidf(self, dataset):
        sents = [e.text_a for e in dataset]
        tv_fit = self.tv.fit_transform(sents)
        self.replace_words = self.tv.get_feature_names_out()
        self.tfidf = tv_fit.toarray()
        return np.argsort(-self.tfidf, axis=-1)


    def perturb(self, text):
        words = text.split()
        m = int(len(words) * self.swap_ratio)
        piece = np.random.choice(self.tfidf.shape[0])
        swap_pos = np.random.randint(0, len(words), m)
        candidate = []
        for i, j in enumerate(swap_pos):
            words[j] = self.replace_words[self.tfidf_idx[piece][i]]
            candidate.append(words[j])
        return " ".join(words)


    def cal_entropy(self, model, dataset):
        perturbed_data = []
        for idx, example in enumerate(dataset):
            for _ in range(self.repeat):
                e = copy.deepcopy(example)
                e.text_a = self.perturb(e.text_a)
                perturbed_data.append(e)
        
        logging.info("There are {} perturbed sentences".format(len(perturbed_data)))
        dataloader = get_dataloader(perturbed_data, batch_size=self.batch_size)
        model.eval()
        probs = []
        for batch in tqdm(dataloader):
            with torch.no_grad():
                inputs, labels = model.process(batch)
                outputs = model(inputs)
            probs.extend(torch.softmax(outputs.logits, dim=-1).cpu().tolist())

        probs = np.array(probs)
        entropy = - np.sum(probs * np.log2(probs), axis=-1)
        entropy = np.reshape(entropy, (self.repeat, -1))
        entropy = np.mean(entropy, axis=0)
        return entropy


    def get_filter_rate(self, entropy):
        filter_rate = sum([int(i) for i in (entropy<self.threshold)])/len(entropy)
        return filter_rate
