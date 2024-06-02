import hydra
from omegaconf import DictConfig
import torch.nn as nn

from ViT.src.model.my_ViT import ViT
from ViT.src.config import ViTConfig


def load_model(cfg: ViTConfig) -> nn.Module:
    model = ViT(cfg)
    return model


@hydra.main(config_path="conf", config_name="infefence", version_base="1.2")
def main(cfg: DictConfig):
    model = load_model(cfg)
    print(model)
