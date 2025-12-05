from .downstream.adv_sc_poisoner import AdvSCPoisoner
from .downstream.prompt_sc_poisoner import PromptSCPoisoner
from .downstream.sc_poisoner import SCPoisoner
from .llm.neuba_poisoner_llm import NeuBAPoisonerLLM
from .llm.por_poisoner_llm import PORPoisonerLLM
from .plm.btop_poisoner import BToPPoisoner
from .plm.lm_cleanse_mask_poisoner import LMCleanseMaskPoisoner
from .plm.lm_cleanse_poisoner import LMCleansePoisoner
from .plm.lm_cleanse_vector_poisoner import LMCleanseVectorPoisoner
from .plm.neuba_poisoner import NeuBAPoisoner
from .plm.por_poisoner import PORPoisoner

POISONERS_LIST = {
    "neuba": NeuBAPoisoner,
    "por": PORPoisoner,
    "neuba_llm": NeuBAPoisonerLLM,
    "por_llm": PORPoisonerLLM,
    "btop": BToPPoisoner,
    "lm_cleanse": LMCleansePoisoner,
    "lm_cleanse_mask": LMCleanseMaskPoisoner,
    "lm_cleanse_vector": LMCleanseVectorPoisoner,
    "sc": SCPoisoner,
    "prompt_sc": PromptSCPoisoner,
    "adv_sc": AdvSCPoisoner,
}


def get_poisoner(config):
    poisoner = POISONERS_LIST[config.method](config)
    return poisoner
