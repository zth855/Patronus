import logging

import numpy as np
import torch

from .defender import Defender


class PRUNEDefender(Defender):
    def __init__(self, config):
        super().__init__(config)
        self.prune_ratio = config.prune_ratio

    def rebulid(self, model):
        logging.info("\n======== Prune Defense ========")
        logging.info("pruning ratio: {}".format(self.prune_ratio))
        head_name = [n for n, c in model.plm.named_children()]
        plm_layers = getattr(model.plm, head_name[0])
        layers = plm_layers.encoder.layer
        for i in range(len(layers)):
            weight = layers[i].output.dense.weight.detach().cpu().numpy()
            weight = self.sort_and_prune(weight)
            layers[i].output.dense.weight.data = torch.from_numpy(weight).float().cuda()
        logging.info("======== Prune Finish! ========\n")
        return model

    def sort_and_prune(self, weight):
        w_shape = weight.shape
        weight = weight.reshape(weight.size)

        order = np.argsort(np.abs(weight))
        weight = sorted(weight, key=abs, reverse=False)
        prune_num = int(self.prune_ratio * len(weight))
        weight[:prune_num] = [0 for i in range(prune_num)]

        recovery_weight = np.zeros_like(weight)
        for i in range(len(weight)):
            recovery_weight[order[i]] = weight[i]

        return recovery_weight.reshape(w_shape)
