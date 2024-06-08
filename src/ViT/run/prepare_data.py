"""
MNIST Data Preparation Script

This script prepares the MNIST dataset for training.

"""

import shutil
from pathlib import Path
import hydra
from omegaconf import DictConfig

from torchvision.datasets.mnist import MNIST

from ViT.run.utils import get_transforms
from ViT.src.config import MNISTDataConfig, TransformConfig


def load_MNIST_data(cfg: MNISTDataConfig):
    """
    Load and preprocess the MNIST dataset.

    Args:
        cfg (MNISTDataConfig): Configuration object
        containing dataset parameters.

    Returns:
        None
    """
    processed_dir: Path = Path(cfg.root)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Deleted: {processed_dir}")

    transform = get_transforms(cfg.transform)

    # Load Train Set
    MNIST(
        root=cfg.root,
        train=cfg.train,
        download=cfg.download,
        transform=transform,
    )
    # Load Test Set
    MNIST(
        root=cfg.root, train=False, download=cfg.download, transform=transform
    )


@hydra.main(version_base=None, config_path="conf", config_name="prepare_data")
def main(cfg: DictConfig):
    """
    Main function for preparing the MNIST dataset.

    Args:
        cfg (DictConfig): Hydra configuration object
        containing dataset parameters.

    Returns:
        None
    """
    if cfg.dataset.name == "MNIST":
        cfg = cfg.dataset.mnist
        dataset_config = MNISTDataConfig(
            root=cfg.root,
            train=cfg.train,
            download=cfg.download,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            transform=TransformConfig(
                type=cfg.transform[0].type,
                mean=None,
                std=None,
            ),
        )
        load_MNIST_data(dataset_config)

    else:
        ValueError("Please select a valid dataset")


if __name__ == "__main__":
    main()
