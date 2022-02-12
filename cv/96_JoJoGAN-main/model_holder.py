import threading
import torch
from copy import deepcopy
from model import Discriminator, Generator
import os

model_dict = {}
create_lock = threading.Lock()
device = "cpu"  # 'cuda' or 'cpu'


def get_or_create_pretrained(pretrained):
    if model_dict.get(pretrained) is None:
        create_lock.acquire()  # 单一线程新建模型
        if model_dict.get(pretrained) is None:
            latent_dim = 512
            # Load original generator
            original_generator = Generator(1024, latent_dim, 8, 2).to(device)
            ckpt = torch.load(
                "models/stylegan2-ffhq-config-f.pt",
                map_location=lambda storage, loc: storage,
            )
            original_generator.load_state_dict(ckpt["g_ema"], strict=False)
            # to be finetuned generator
            generator = deepcopy(original_generator)
            ckpt = f"{pretrained}.pt"
            ckpt = torch.load(
                os.path.join("models", ckpt), map_location=lambda storage, loc: storage
            )
            generator.load_state_dict(ckpt["g"], strict=False)

            with torch.no_grad():
                generator.eval()
            model_dict[pretrained] = generator

        create_lock.release()
    return model_dict[pretrained]
