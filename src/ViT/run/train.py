# src/ViT/run/train.py
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


from ViT.src.model.my_ViT import ViT
from ViT.run.prepare_data import load_MNIST_data
from ViT.src.config import (
    ViTConfig,
    TrainConfig,
    TrainerConfig,
    SaveModelConfig,
    DatasetConfig,
    TransformConfig,
    ModelConfig,
)

from ViT.src.datamodule.seg import MNISTDataModule
from ViT.src.modelmodule.seg import ViTModelModule

np.random.seed(0)
torch.manual_seed(0)


def train(train_loader: DataLoader, train: TrainConfig):
    # Initialize model
    model = train.model
    trainer_config = train.trainer
    save_model_config = train.dir.model_dir
    # model = ViT(
    #     vit_config.dimensions,
    #     n_patches=vit_config.n_patches,
    #     n_blocks=vit_config.n_blocks,
    #     hidden_d=vit_config.hidden_d,
    #     n_heads=vit_config.n_heads,
    #     out_d=vit_config.out_d,
    # ).to(trainer_config.device)

    # Initialize optimizer and loss function
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

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(trainer_config.device, desc="Testing"):
            x, y = batch
            x, y = x.to(trainer_config.device), y.to(trainer_config.device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(
                trainer_config.device
            )

            correct += (
                torch.sum(torch.argmax(y_hat, dim=1) == y)
                .detach()
                .cpu()
                .item()
            )
            total += len(x)

        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")

    # Save the models
    if not os.path.exists(save_model_config.model_path):
        os.makedirs(save_model_config.model_path)

    dir_save = Path(save_model_config.model_path) / Path(
        save_model_config.model_name
    )
    torch.save(model, dir_save)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: TrainConfig):
    print(cfg.ViT)
    datamodule = MNISTDataModule(cfg)
    model = ViTModelModule(cfg)
    print(model.param)
    # print(model)
    # train(
    #     train_loader=datamodule.train_loader,train=
    # )
    # # Get the current date
    # current_date = datetime.now().strftime("%Y-%m-%d %H-%M")

    # # Update the model_name with the current date
    # save_model_config = SaveModelConfig(
    #     model_name=f"{cfg.save_model.model_name}_{current_date}",
    #     model_path=cfg.save_model.model_path,
    # )

    # # Extract variables from cfg and map them to data classes
    # vit_config = ViTConfig(
    #     dimensions=cfg.ViT.dimensions,
    #     n_patches=cfg.ViT.n_patches,
    #     n_blocks=cfg.ViT.n_blocks,
    #     hidden_d=cfg.ViT.hidden_d,
    #     n_heads=cfg.ViT.n_heads,
    #     out_d=cfg.ViT.out_d,
    # )

    # trainer_config = TrainerConfig(
    #     device=cfg.trainer.device,
    #     LR=cfg.trainer.LR,
    #     N_EPOCHS=cfg.trainer.N_EPOCHS,
    #     loss_step=cfg.trainer.loss_step,
    #     batch_size=cfg.trainer.batch_size,
    #     loss_step=cfg.trainer.loss_step,
    # )

    # if cfg.dataset.name == "MNIST":
    #     dataset_config = DatasetConfig(
    #         name=cfg.dataset.mnist.name,
    #         root=cfg.dataset.root,
    #         train=cfg.dataset.mnist.train,
    #         download=cfg.dataset.mnist.download,
    #         batch_size=cfg.dataset.mnist.batch_size,
    #         shuffle=cfg.dataset.mnist.shuffle,
    #         transform=TransformConfig(
    #             type=cfg.dataset.mnist.transform[0].type,
    #             mean=None,
    #             std=None,
    #         ),
    #     )
    #     # Load data
    #     train_loader, test_loader = load_MNIST_data(dataset_config)

    # # Call the train function with extracted parameters
    # train(
    #     train_loader,
    #     trainer_config.device,
    #     vit_config,
    #     trainer_config,
    #     save_model_config,
    # )


if __name__ == "__main__":
    main()
