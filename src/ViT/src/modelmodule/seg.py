from ViT.src.config import ViTConfig, TrainConfig
from ViT.src.model import my_ViT as ViT


class ViTModelModule(ViTConfig):
    def __init__(self, cfg: TrainConfig):
        super().__init__(
            dimensions=cfg.ViT.dimensions,
            n_patches=cfg.ViT.n_patches,
            n_blocks=cfg.ViT.n_blocks,
            hidden_d=cfg.ViT.hidden_d,
            n_heads=cfg.ViT.n_heads,
            out_d=cfg.ViT.out_d,
        )
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = ViT(self.param)
            return self._model
