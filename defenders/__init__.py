from .onion import ONIONDefender
from .rap import RAPDefender
from .strip import STRIPDefender
from .rap_m import RAP_PLM_Defender
from .strip_m import STRIP_PLM_Defender
from .reinit_mlp import REINIT_MLP_Defender
from .prune import PRUNEDefender
from .recipe import RECIPEDefender
from .lm_cleanse import LM_CLEANSE
from .lm_cleanse_llm import LM_CLEANSE_LLM
from .lm_cleanse_mask import LM_CLEANSE_MASK
from .lm_cleanse_vector import LM_CLEANSE_VECTOR
from .lm_sanitator import LM_SANITATOR
from .lm_sanitator_mask import LM_SANITATOR_MASK
from .lm_sanitator_vector import LM_SANITATOR_VECTOR
from .reinit_t_embeds import Reinit_Trigger_Embeds_Defender
from .btu import BTUDefender


DEFENDER_LIST = {
    'lm_cleanse': LM_CLEANSE,
    'lm_cleanse_llm': LM_CLEANSE_LLM,
    'lm_cleanse_mask': LM_CLEANSE_MASK,
    'lm_cleanse_vector': LM_CLEANSE_VECTOR,
    'lm_sanitator': LM_SANITATOR,
    'lm_sanitator_mask': LM_SANITATOR_MASK,
    'lm_sanitator_vector': LM_SANITATOR_VECTOR,
    'onion': ONIONDefender,      
    'rap': RAPDefender,
    'strip': STRIPDefender,  
    'rap_m': RAP_PLM_Defender,
    'strip_m': STRIP_PLM_Defender,  
    'prune': PRUNEDefender,
    'recipe': RECIPEDefender,
    'reinit_mlp': REINIT_MLP_Defender, 
    'reinit_t_embeds': Reinit_Trigger_Embeds_Defender,
    'btu': BTUDefender
}


def get_defender(config):
    defender = DEFENDER_LIST[config.type](config)
    return defender