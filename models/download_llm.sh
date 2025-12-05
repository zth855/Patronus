# GPT-Neo-1.3B
# GPT2 XL 1.5B
# GPT2 Large 774M

huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir /data/fangly/models/bert-base-uncased --local-dir-use-symlinks False
huggingface-cli download --resume-download EleutherAI/gpt-neo-1.3B --local-dir /data/fangly/models/gpt-neo-1.3b --local-dir-use-symlinks False
huggingface-cli download --resume-download openai-community/gpt2-xl --local-dir /data/fangly/models/gpt2-xl --local-dir-use-symlinks False
huggingface-cli download --resume-download openai-community/gpt2-large --local-dir /data/fangly/models/gpt2-large --local-dir-use-symlinks False