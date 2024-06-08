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

# def load_model(cfg: InferenceConfig) -> nn.Module:
#     model = ViT(cfg.model)
#     model.(torch.load(cfg.model_path))
#     return model


def inference(loader: DataLoader, inf: InferenceConfig):
    model = torch.load(inf.model_path)
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
    if not os.path.exists(inf.save):
        os.makedirs(inf.save)

    dir_save = Path(inf.save) / Path(inf.model_name)
    df.to_csv(str(dir_save) + ".csv", index=False)

    return df


@hydra.main(config_path="conf", config_name="inference", version_base="1.2")
def main(cfg: DictConfig):

    if cfg.dataset.name == "MNIST":
        datamodule = MNISTDataModule(cfg.dataset.mnist)
        loader = datamodule.test_loader

    if cfg.model.name == "ViT":
        cfg = cfg.model.ViT

        model_name = Path(cfg.model_path).name
        inf = InferenceConfig(
            model_name=model_name,
            model_path=cfg.model_path,
            device=cfg.device,
            save=cfg.save,
        )

    pred = inference(loader, inf)
    print(pred)


if __name__ == "__main__":
    main()
