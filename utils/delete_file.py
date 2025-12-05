import os

path = '../results/defense/adv_pretrain'
for root, dirs, files in os.walk(path):
    for name in dirs:
        dir_path = os.path.join(root, name)
        for f in os.listdir(dir_path):
            if f == 'finetune_model.ckpt' or f == 'pretrain_model.ckpt':
                file_path = os.path.join(dir_path, f)
                os.remove(file_path)
        # os.removedirs(dir_path)