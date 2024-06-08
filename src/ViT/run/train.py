"""
Trainer Script

This script trains a model using the provided
configuration and saves the trained model.

"""

import hydra
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import os
from pathlib import Path
from datetime import datetime
import numpy as np

from ViT.src.config import (
    TrainConfig,
    TrainerConfig,
    SaveModelConfig,
)

from ViT.src.datamodule.seg import MNISTDataModule
from ViT.src.modelmodule.seg import ViTModelModule

np.random.seed(0)
torch.manual_seed(0)


def training(train_loader: DataLoader, train: TrainConfig):
    """
    Training Function

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        train (TrainConfig): Configuration for training.

    """
    # Initialize model
    model = train.model.model_config
    trainer_config = train.trainer
    save_model_config = train.dir

    optimizer = Adam(model.parameters(), lr=trainer_config.LR)
    criterion = CrossEntropyLoss()
    # Training loop
    for epoch in trange(trainer_config.N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for i, batch in enumerate(
            tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1} in training",
                leave=False,
            )
        ):
            x, y = batch
            x, y = x.to(trainer_config.device), y.to(trainer_config.device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i % trainer_config.loss_step) == 0:
                print(f"\n Step {i} loss: {train_loss}\n")

        print(
            f"Epoch {epoch + 1}/{trainer_config.N_EPOCHS}\
            loss: {train_loss:.2f}\n"
        )

    # # Test loop
    # with torch.no_grad():
    #     correct, total = 0, 0
    #     test_loss = 0.0
    #     for batch in tqdm(trainer_config.device, desc="Testing"):
    #         x, y = batch
    #         x, y = x.to(trainer_config.device), y.to(trainer_config.device)
    #         y_hat = model(x)
    #         loss = criterion(y_hat, y)
    #         test_loss += loss.detach().cpu().item() / len(
    #             trainer_config.device
    #         )

    #         correct += (
    #             torch.sum(torch.argmax(y_hat, dim=1) == y)
    #             .detach()
    #             .cpu()
    #             .item()
    #         )
    #         total += len(x)

    #     print(f"Test loss: {test_loss:.2f}")
    #     print(f"Test accuracy: {correct / total * 100:.2f}%")

    # Save the models
    if not os.path.exists(save_model_config.model_path):
        os.makedirs(save_model_config.model_path)

    dir_save = Path(save_model_config.model_path) / Path(
        save_model_config.model_name
    )
    torch.save(model, dir_save)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: TrainConfig):
    """
    Main Function

    Args:
        cfg (TrainConfig): Configuration object containing training parameters.

    """

    date_time = datetime.now().strftime("%Y-%m-%d-%H_%M")
    if cfg.dataset.name == "MNIST":
        datamodule = MNISTDataModule(cfg.dataset.mnist)

    if cfg.model.name == "ViT":
        modelmodule = ViTModelModule(cfg.model.ViT)
        save_model_config = SaveModelConfig(
            model_path=cfg.model.ViT.model_path,
            model_name=Path(cfg.model.name + "_" + date_time),
        )

    trainer = TrainerConfig(**cfg.trainer)
    train = TrainConfig(modelmodule, trainer, save_model_config)

    # Training of the model
    training(datamodule.train_loader, train)


if __name__ == "__main__":
    main()
