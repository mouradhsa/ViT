"""
Module: seg.py

This module contains the definition of the MNISTDataModule class,
which is responsible for creating data loaders for the MNIST dataset.

"""

from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from ViT.src.config import TrainConfig, MNISTDataConfig
from ViT.run.utils import get_transforms


# Define the custom dataset class
class MNISTWithoutLabels(MNIST):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)  # Discard the label
        return img


class MNISTDataModule(MNISTDataConfig):
    """
    MNISTDataModule class creates data loaders for the MNIST dataset.

    Args:
        cfg (TrainConfig): Configuration object containing dataset parameters.

    Attributes:
        cfg (TrainConfig): Configuration object containing dataset parameters.
        transform (callable): The data transformation function.
        _train_loader (DataLoader): DataLoader object for the training dataset.
        _test_loader (DataLoader): DataLoader object for the testing dataset.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        """
        Initializes the MNISTDataModule with the provided configuration.

        Args:
            cfg (TrainConfig): Configuration object containing
            dataset parameters.
        """
        super().__init__(
            root=cfg.root,
            train=cfg.train,
            download=cfg.download,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            transform=cfg.transform,
        )
        self.cfg = cfg
        self.transform = get_transforms(self.transform[0])
        self._train_loader = None
        self._test_loader = None

    @property
    def train_loader(self):
        """
        Property: train_loader

        Creates and returns a DataLoader object for the training dataset.

        Returns:
            DataLoader: DataLoader object for the training dataset.
        """
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
        """
        Property: test_loader

        Creates and returns a DataLoader object for the testing dataset.


        Returns:
            DataLoader: DataLoader object for the testing dataset.
        """
        if self._test_loader is None:
            test_dataset = MNISTWithoutLabels(
                root=self.root,
                train=False,
                download=self.download,
                transform=self.transform,
            )
            self._test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False
            )
        return self._test_loader
