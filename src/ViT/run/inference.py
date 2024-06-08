"""
Inference Script

This script performs inference using a trained model and saves
the predictions to a CSV file.

"""

import hydra
from omegaconf import DictConfig
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from ViT.src.config import InferenceConfig
from ViT.src.datamodule.seg import MNISTDataModule


def inference(loader: DataLoader, inf: InferenceConfig):
    """
    Perform inference using the provided DataLoader and configuration.

    Args:
        loader (DataLoader): DataLoader object for inference.
        inf (InferenceConfig): Configuration object
        containing inference parameters.

    Returns:
        None
    """
    # Load the model
    model_path = Path(inf.model_dir) / Path(inf.model_name)
    model = torch.load(model_path)
    model.to(inf.device)
    model.eval()

    # Store predictions and probabilities
    all_predictions = []
    all_probabilities = []
    all_indices = []
    with torch.no_grad():
        for i, x in enumerate(tqdm(loader, desc="Inference")):
            x = x.to(inf.device)
            y_hat = model(x)

            # Get the predicted labels and probabilities
            probabilities = torch.softmax(y_hat, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)

            all_probabilities.extend(probabilities.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_indices.extend(
                range(i * loader.batch_size, i * loader.batch_size + x.size(0))
            )

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "index": all_indices,
            "prediction": all_predictions,
            "probability": [prob.tolist() for prob in all_probabilities],
        }
    )

    # Save predictions to CSV
    if not os.path.exists(inf.save):
        os.makedirs(inf.save)

    dir_save = Path(inf.save) / Path(inf.model_name)
    df.to_csv(str(dir_save) + ".csv", index=False)


@hydra.main(config_path="conf", config_name="inference", version_base="1.2")
def main(cfg: DictConfig):
    """
    Main function for inference.

    Args:
        cfg (DictConfig): Hydra configuration
        object containing inference parameters.

    Returns:
        None
    """
    if cfg.dataset.name == "MNIST":
        datamodule = MNISTDataModule(cfg.dataset.mnist)
        loader = datamodule.test_loader

    if cfg.model.name == "ViT":
        cfg = cfg.model.ViT

        inf = InferenceConfig(
            model_name=cfg.model_name,
            model_dir=cfg.models_dir,
            device=cfg.device,
            save=cfg.save,
        )

    # Perform inference
    inference(loader, inf)


if __name__ == "__main__":
    main()
