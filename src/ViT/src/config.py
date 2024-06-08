"""
This module defines configuration classes for various aspects of the
machine learning pipeline using dataclasses.
These configurations include model settings, training parameters, dataset
parameters, and more.

Classes:
    SaveModelConfig: Configuration for saving the model.
    ViTConfig: Configuration for the Vision Transformer (ViT) model.
    DirConfig: Configuration for directories used in the project.
    ModelConfig: General configuration for a machine learning model.
    TrainerConfig: Configuration for the training process.
    TrainConfig: Configuration for the overall training setup.
    TransformConfig: Configuration for data transformation.
    DatasetConfig: Configuration for the dataset.
"""

from dataclasses import dataclass
from typing import Optional, Any, List
from pathlib import Path


@dataclass
class SaveModelConfig:
    """
    Configuration for saving the model.

    Attributes:
        model_name (str): The name of the model.
        model_path (str): The path where the model will be saved.
    """

    model_name: str
    model_path: str


@dataclass
class ModelConfig:
    """
    General configuration for a machine learning model.

    Attributes:
        name (str): The name of the model.
        param (dict[str, Any]): A dictionary of parameters for the model.
    """

    name: str
    param: dict[str, Any]


@dataclass
class ViTConfig(ModelConfig):
    """
    Configuration for the Vision Transformer (ViT) model.

    Attributes:
        dimensions (int): The input dimension size.
        n_patches (int): The number of patches the image is divided into.
        n_blocks (int): The number of transformer blocks.
        hidden_d (int): The hidden dimension size.
        n_heads (int): The number of attention heads.
        out_d (int): The output dimension size.
    """

    dimensions: List[int]
    n_patches: int
    n_blocks: int
    hidden_d: int
    n_heads: int
    out_d: int

    def __init__(
        self,
        dimensions: List[int],
        n_patches: int,
        n_blocks: int,
        hidden_d: int,
        n_heads: int,
        out_d: int,
    ):
        super().__init__(
            name="ViT",
            param={
                "dimensions": dimensions,
                "n_patches": n_patches,
                "n_blocks": n_blocks,
                "hidden_d": hidden_d,
                "n_heads": n_heads,
                "out_d": out_d,
            },
        )


@dataclass
class TrainerConfig:
    """
    Configuration for the training process.

    Attributes:
        device (str): The device to be used for training
        (e.g., 'cpu' or 'cuda').
        LR (float): The learning rate for training.
        N_EPOCHS (int): The number of epochs for training.
        loss_step (int): The step interval for logging loss.
        device (int): The GPU device ID.
    """

    device: str
    LR: float
    N_EPOCHS: int
    loss_step: int


@dataclass
class TrainConfig:
    """
    Configuration for the overall training setup.

    Attributes:
        model (ModelConfig): The model configuration.
        trainer (TrainerConfig): The training configuration.
        dir (DirConfig): The directory configuration.
    """

    model: ModelConfig
    trainer: TrainerConfig
    dir: SaveModelConfig


@dataclass
class InferenceConfig:
    model_name: str
    model_path: Path
    device: str
    save: Path


@dataclass
class TransformConfig:
    """
    Configuration for data transformation.

    Attributes:
        type (str): The type of transformation.
        mean (Optional[float]): The mean value for normalization.
        std (Optional[float]): The standard deviation for normalization.
    """

    type: str
    mean: Optional[float]
    std: Optional[float]


@dataclass
class DatasetConfig:
    """
    Base configuration for datasets.

    Attributes:
        batch_size (int): The batch size for data loading.
        transform (TransformConfig): The data transformation configuration.
        root (str): The root directory of the dataset.
        shuffle (bool): Whether to shuffle the dataset.
    """

    batch_size: int
    transform: TransformConfig
    root: str
    shuffle: bool


@dataclass
class MNISTDataConfig(DatasetConfig):
    """
    Configuration for the MNIST dataset.

    Attributes:
        train (bool): Whether the dataset is for training.
        download (bool): Whether to download the dataset if not present.
    """

    train: bool
    download: bool
