from torch.utils.data import Dataset, DataLoader
import cv2

class AlbumentationsDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"画像がありません: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        
        return img, label
    

def AlbumentationsImageDataset(train_paths, train_labels, val_paths, val_labels, train_transform, val_transform):
    train_dataset = AlbumentationsDataset(train_paths, train_labels, train_transform)
    val_dataset   = AlbumentationsDataset(val_paths, val_labels, val_transform)
    return train_dataset, val_dataset


def build_data_loaders(train_dataset, val_dataset, batch_size, num_workers=2, pin_memory=True):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader

