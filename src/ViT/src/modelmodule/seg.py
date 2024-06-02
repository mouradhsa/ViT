from ViT.src.config import ViTConfig, TrainConfig
from ViT.src.model.my_ViT import ViT


class ViTModelModule(ViTConfig):
    def __init__(self, cfg: TrainConfig):
        super().__init__(
            dimensions=cfg.dimensions,
            n_patches=cfg.n_patches,
            n_blocks=cfg.n_blocks,
            hidden_d=cfg.hidden_d,
            n_heads=cfg.n_heads,
            out_d=cfg.out_d,
        )
        self._model_config = None

    @property
    def model_config(self):
        if self._model_config is None:
            self._model_config = ViT(**self.param)
            return self._model_config
