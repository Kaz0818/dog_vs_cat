import yaml
from pathlib import Path
from typing import List, Tuple
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    
    
def enumerate_and_split_imagefolder(
) -> Tuple[Tuple[List[str]], List[int], Tuple[List[str]], List[int], List[str]]:
    
    """ 
    ImageFolderから画像パスとラベルをtrain/valに分割する
    
    Returns:
        (train_paths, train_labels), (val_paths, val_labels), class_names
    """
    data_root = PROJECT_ROOT / config["data_root"]
    
    image_folder = ImageFolder(str(data_root))
    samples = image_folder.samples
    class_names = image_folder.classes
    
    all_paths = [p for p,_ in samples]
    all_labels = [y for _,y in samples]
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=config["validation_split"],
        stratify=all_labels,
        random_state=config["random_state"]
    )
    return (train_paths, train_labels), (val_paths, val_labels), class_names

if __name__ == "__main__":
    (train_paths, train_labels), (val_paths, val_labels), class_names = enumerate_and_split_imagefolder()
    print(f"classes: {class_names}")
    print(f"train_path:{len(train_paths)}, val: {len(val_paths)}")
    