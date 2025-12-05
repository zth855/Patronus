import logging

import torch

from .defender import Defender


class REINIT_MLP_Defender(Defender):
    def __init__(self, config):
        super().__init__(config)
        self.reinit_ll = config.reinit_ll
        self.reinit_pl = config.reinit_pl

    def rebulid(self, model):
        logging.info("\n======== Reinit Defense ========")

        head_name = [n for n, c in model.plm.named_children()]
        plm_layers = getattr(model.plm, head_name[0])
        if self.reinit_ll:
            weight = plm_layers.encoder.layer[-1].output.dense.weight.data
            plm_layers.encoder.layer[-1].output.dense.weight.data = (
                torch.nn.init.kaiming_normal_(weight).float().cuda()
            )
        if self.reinit_pl:
            weight = plm_layers.pooler.dense.weight.data
            plm_layers.pooler.dense.weight.data = (
                torch.nn.init.kaiming_normal_(weight).float().cuda()
            )

        logging.info("======== Reinit Finish! ========\n")
        return model
