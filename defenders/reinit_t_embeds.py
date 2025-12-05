from .defender import Defender
import logging
import torch


class Reinit_Trigger_Embeds_Defender(Defender):
    def __init__(self, config):
        super().__init__(config)
        self.triggers = config.triggers

    def rebulid(self, model):
        logging.info("\n======== Word Embeddings Reinit Defense ========")
        for trigger_id in model.token_to_id(self.triggers):
            embeddings = model.word_embedding().weight[trigger_id, :].data 
            model.word_embedding().weight[trigger_id, :].data  = torch.rand_like(embeddings).float().cuda()
        logging.info("======== Word Embeddings Reinit Finish! ========\n")
        return model


