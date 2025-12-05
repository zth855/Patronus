import logging

from .llm import LLMVictim
from .mlm import MLMVictim
from .plm import PLMVictim
from .sc import SCVictim

VICTIM_LIST = {
    "plm": PLMVictim,  # Model
    "mlm": MLMVictim,  # MaskedLM
    "sc": SCVictim,  # SequenceClassification
    "llm": LLMVictim,  # Large Language Model
}


def get_victim(config):
    logging.info("\n> Loading {} from {} <\n".format(config.type, config.path))
    victim = VICTIM_LIST[config.type](config)
    return victim
