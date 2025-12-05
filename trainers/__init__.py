from .downstream.finetune_sc_trainer import FineTuneSCTrainer
from .downstream.prompt_sc_trainer import PromptSCTrainer
from .llm.neuba_llm_trainer import NeuBA_LLM_Trainer
from .llm.por_llm_trainer import POR_LLM_Trainer
from .plm.adv_trainer import ADVTrainer
from .plm.btop_trainer import BToPTrainer
from .plm.neuba_adapt_trainer import NeuBA_ADAPT_Trainer
from .plm.neuba_trainer import NeuBATrainer
from .plm.por_adapt_trainer import POR_ADAPT_Trainer
from .plm.por_trainer import PORTrainer

TRAINERS_LIST = {
    "neuba": NeuBATrainer,
    "neuba_adapt": NeuBA_ADAPT_Trainer,
    "por": PORTrainer,
    "por_adapt": POR_ADAPT_Trainer,
    "neuba_llm": NeuBA_LLM_Trainer,
    "por_llm": POR_LLM_Trainer,
    "btop": BToPTrainer,
    "adv": ADVTrainer,
    "finetune_sc": FineTuneSCTrainer,
    "prompt": PromptSCTrainer,
}


def get_trainer(config, save_dir):
    trainer = TRAINERS_LIST[config.method](config, save_dir)
    return trainer
