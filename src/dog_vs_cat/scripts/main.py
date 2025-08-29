import json, shutil, datetime
from pathlib import Path
import yaml
import random
from zoneinfo import ZoneInfo
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dog_vs_cat.transforms.image_transforms import build_image_transforms
from src.dog_vs_cat.split_data.split_train_val import enumerate_and_split_imagefolder
from src.dog_vs_cat.dataset_data_loader.make_dataloader import AlbumentationsImageDataset, build_data_loaders
from src.dog_vs_cat.engine.trainer import Trainer
from src.dog_vs_cat.models.build_model import build_model
from src.dog_vs_cat.optimizers.build_optimizer import build_optimizer
from src.dog_vs_cat.visualization.visualizer import Visualizer

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

# 1) YAMLをロードしてPythonのdictに変換
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2) 再現性のためのシード固定（DataLoaderのworkerにも効かせる）
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPSは manual_seed_all は不要（torch.manual_seed でOK）

def worker_init_fn(worker_id: int):
    # 各workerの乱数を安定化（NumPy/Pythonは32bit範囲に丸める）
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def main():
    print(f"device: {device}")
    # 1) シード固定（YAMLは random_seed で統一）
    seed_everything(config["random_state"])

    # 2) 変換
    train_transform, val_transform = build_image_transforms(
        image_size=config["image_size"],
        mean=tuple(config["mean"]),
        std=tuple(config["std"]),
    )

    # 3) 列挙→層化分割→Dataset
    (train_paths, train_labels), (validation_paths, validation_labels), class_names = enumerate_and_split_imagefolder()
        
    train_dataset, validation_dataset = AlbumentationsImageDataset(train_paths, train_labels, validation_paths, validation_labels, train_transform, val_transform)
    train_loader, val_loader = build_data_loaders(train_dataset, validation_dataset, config["batch_size"], num_workers=2)

    # モデル
    model = build_model(config["model_name"], num_classes=len(class_names))
    
    # 損失関数
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = build_optimizer(
        model,
        name=str(config.get("optimizer", "adamw")),
        lr=float(config.get("learning_rate", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    
    ts = datetime.datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{ts}_{config['model_name']}_bs{config['batch_size']}_lr{config['learning_rate']}_ep{config['epochs']}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    
    # 学習ループ
    trainer = Trainer(model,
                      train_loader,
                      val_loader,
                      criterion,
                      optimizer, config['epochs'],
                      device=device,
                      checkpoint_dir=str(run_dir / "checkpoints" ),
)
    history = trainer.train()
    
    shutil.copy("configs/config.yaml", run_dir / "config.yaml")
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({
            "val_acc": float(history['val_acc'][-1]),
            "val_loss": float(history['val_loss'][-1]),
            "train_acc": float(history['train_acc'][-1]),
            "train_loss": float(history['train_loss'][-1]),
        }, f, indent=2)
    
    vis = Visualizer()
    vis.metrics_plot(
        history["train_loss"], history["val_loss"],
        history["train_acc"], history["val_acc"],
        save_path=run_dir / "results" / "train_val_loss_plot.png"
    )
    
    vis.plot_confusion_matrix_display(
        model, val_loader, class_names, device,
        cm_save_path=run_dir / "results" / "confusion_matrix.png"
    )
    
    vis.plot_misclassified_images(
        model, val_loader, class_names, device,
        save_path=run_dir / "results"/ "misclassified_images.png",
        max_images=16
    )
    
    vis.result_classification_report(
        model, val_loader, class_names, device
    )
        

if __name__ == "__main__":
    main()
