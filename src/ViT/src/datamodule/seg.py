from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST


from ViT.src.config import TrainConfig, MNISTDataConfig
from ViT.run.utils import get_transforms


class MNISTDataModule(MNISTDataConfig):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__(
            batch_size=cfg.dataset.mnist.batch_size,
            transform=cfg.dataset.mnist.transform,
            root=cfg.dataset.mnist.root,
            shuffle=cfg.dataset.mnist.shuffle,
            train=cfg.dataset.mnist.train,
            download=cfg.dataset.mnist.download,
        )
        self.cfg = cfg

        self._train_loader = None
        self._test_loader = None

    @property
    def train_loader(self):
        if self._train_loader is None:
            train_dataset = MNIST(
                root=self.root,
                train=True,
                download=self.download,
                transform=self.transform,
            )
            self._train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
            )
        return self._train_loader

    @property
    def test_loader(self):
        if self._test_loader is None:
            test_dataset = MNIST(
                root=self.root,
                train=False,
                download=self.download,
                transform=self.transform,
            )
            self._test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )
        return self._test_loader
