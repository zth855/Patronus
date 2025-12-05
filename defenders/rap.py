import copy
import logging
import random

import numpy as np
import torch

from .defender import Defender
from .utils.dataloader import get_dataloader


class RAPDefender(Defender):
    def __init__(self, config):
        super().__init__(config)
        self.epochs = config.epochs  # Number of RAP training epochs. Default to 5.
        self.batch_size = config.batch_size
        self.lr = float(
            config.lr
        )  # Learning rate for RAP triggers embeddings. Default to 1e-2.
        self.trigger = (
            config.trigger
        )  # The triggers to insert in texts. Default to `["cf"]`.
        self.protect_label = None
        self.prob_range = (
            config.prob_range
        )  # The upper and lower bounds for probability change. Default to `[-0.1, -0.3]`.
        self.scale = config.scale  # Scale factor for RAP loss. Default to 1.
        self.frr = (
            config.frr
        )  # Allowed false rejection rate on clean dev dataset. Default to 0.01.

    def filter(self, model, clean_dataset, poison_dataset):
        logging.info("\n======== Rap Defense ========")
        clean_data = poison_dataset.pop("test-clean")
        self.protect_label = int(list(poison_dataset.keys())[0].split("-")[3])
        logging.info("protected label: {}".format(self.protect_label))
        poison_data_dict = {}
        for key in poison_dataset.keys():
            if int(key.split("-")[3]) == self.protect_label:
                poison_data_dict[key] = poison_dataset[key]

        clean_dev = []
        for example in clean_dataset["dev"]:
            if example.label == self.protect_label:
                clean_dev.append(example)

        model.eval()
        self.model = model
        self.prepare_trigger_embedding(self.model)
        self.construct(clean_dev)

        clean_dev_prob = self.rap_prob(self.model, clean_dev)
        self.threshold = np.nanpercentile(clean_dev_prob, self.frr * 100)
        logging.info(
            "Constrain FRR to {}, threshold = {}".format(self.frr, self.threshold)
        )

        clean_prob = self.rap_prob(self.model, clean_data, clean=False)
        filter_rate = self.get_filter_rate(clean_prob)
        logging.info(
            "Test-clean diff: {}, filter rate: {}".format(
                np.mean(clean_prob), filter_rate
            )
        )

        for key, poison_data in poison_data_dict.items():
            poison_prob = self.rap_prob(self.model, poison_data, clean=False)
            filter_rate = self.get_filter_rate(poison_prob)
            logging.info(
                "{} diff: {}, filter rate: {}".format(
                    key, np.mean(poison_prob), filter_rate
                )
            )

        logging.info("======== Rap Finish ========\n")

    def prepare_trigger_embedding(self, model):
        embeddings = model.word_embedding().weight
        self.trigger_id = model.token_to_id(self.trigger)
        self.norm = (
            embeddings[self.trigger_id, :].view(1, -1).to(model.device).norm().item()
        )

    def construct(self, clean_dev):
        logging.info("------ construct model ------")
        rap_dev = self.rap_poison(clean_dev)
        dataloader = get_dataloader(clean_dev, self.batch_size)
        rap_dataloader = get_dataloader(rap_dev, self.batch_size)
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct_num = 0
            for batch, rap_batch in zip(dataloader, rap_dataloader):
                prob = self.get_output_prob(self.model, batch)
                rap_prob = self.get_output_prob(self.model, rap_batch)
                loss, correct = self.rap_iter(prob, rap_prob)
                epoch_loss += loss * len(batch)
                correct_num += correct
            epoch_loss /= len(clean_dev)
            asr = correct_num / len(clean_dev)
            logging.info(
                "Epoch: {}, RAP loss: {:.4}, success rate {:.4}".format(
                    epoch + 1, epoch_loss, asr
                )
            )

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
        loss = self.scale * torch.mean(
            (diff > self.prob_range[0]) * (diff - self.prob_range[0])
        ) + torch.mean((diff < self.prob_range[1]) * (self.prob_range[1] - diff))
        correct = ((diff < self.prob_range[0]) * (diff > self.prob_range[1])).sum()
        loss.backward()

        weight = self.model.word_embedding().weight
        grad = weight.grad
        weight.data[self.trigger_id, :] -= self.lr * grad[self.trigger_id, :]
        weight.data[self.trigger_id, :] *= (
            self.norm / weight.data[self.trigger_id, :].norm().item()
        )
        del grad

        return loss.item(), correct

    def rap_prob(self, model, data, clean=True):
        model.eval()
        rap_data = self.rap_poison(data)
        dataloader = get_dataloader(data, self.batch_size)
        rap_dataloader = get_dataloader(rap_data, self.batch_size)
        prob_diffs = []

        with torch.no_grad():
            for batch, rap_batch in zip(dataloader, rap_dataloader):
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

    def get_filter_rate(self, prob):
        filter_rate = sum([int(i) for i in (prob < self.threshold)]) / len(prob)
        return filter_rate
