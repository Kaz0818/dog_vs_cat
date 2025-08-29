import os
import io
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm 

class Visualizer: 
    def __init__(self, writer=None, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
        self.writer = writer
        self.mean = mean
        self.std = std

    def _denormalize(self, img: torch.Tensor, mean, std) -> torch.Tensor:
        """
        img: [C,H,W] (0〜1で標準化済みのテンソル。ToTensorV2後)
        mean, std: 長さ3の(リスト/タプル)
        戻り値: [C,H,W] (0〜1範囲にクリップ)
        """
        if not torch.is_tensor(img):
            img = torch.tensor(img)
        mean = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1,1,1)
        std  = torch.tensor(std,  dtype=img.dtype, device=img.device).view(-1,1,1)
        img = img * std + mean
        return img.clamp(0, 1)
        
# ---------Train Loss VS Validation Loss Plot--------------------------
    def metrics_plot(self, train_losses, val_losses, train_accuracies, val_accuracies):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        axes[0].plot(train_losses, label='Train', lw=3)
        axes[0].plot(val_losses, label='Val', lw=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Loss Curve')
        axes[1].plot(train_accuracies, label='Train Acc', lw=3)
        axes[1].plot(val_accuracies, label='Val Acc', lw=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].set_title('Accuracy Curve')
        plt.tight_layout()
        plt.show()

    
      
    
    # -----------------Confusion Matrix Plot----------------------------                                 
    def plot_confusion_matrix_display(self, model, dataloader, class_names, device,
                                    normalize=True, cm_save_path=None, epoch=None):
        
        model.to(device) # <--- model を device に送るのを追加
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for X, y in tqdm(dataloader, desc="Generating CM", leave=False):
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        
        cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(12, 8))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
        plt.title("Confusion Matrix (Normalized)" if normalize else "Confusion Matrix")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # --- TensorBoardに画像として追加 ---
        if self.writer:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
            image_np = np.array(image)
            image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0
            tag = 'Confusion_Matrix_Display'
            step = epoch if epoch is not None else 0
            self.writer.add_image(tag, image_tensor, global_step=step)
        
        if cm_save_path:            
            plt.savefig(cm_save_path)
            print(f"[INFO] Saved confusion matrix image to {cm_save_path}")
        else:
            plt.show()
            
        plt.close()
        
    # ---------------検証データで間違えたものだけPlotする=------------------------
    def plot_misclassified_images(self, model, dataloader, class_names, device, # model, dataloader, device を追加
                                  max_images=25):
        model.to(device)
        model.eval()
        mis_imgs, mis_preds, mis_trues = [], [], []
    
        with torch.no_grad():
            for X_val, y_val in tqdm(dataloader, desc="Finding Misclassified", leave=False):
                X_val, y_val = X_val.to(device), y_val.to(device)
                preds = model(X_val).argmax(dim=1)
                wrong = preds != y_val
                if wrong.any():
                    mis_imgs.extend(X_val[wrong].cpu())
                    mis_preds.extend(preds[wrong].cpu())
                    mis_trues.extend(y_val[wrong].cpu())
                    if len(mis_imgs) >= max_images:
                        break
    
        n_images = min(len(mis_imgs), max_images)
        if n_images == 0:
            print("[INFO] No misclassified images found.")
            return 
        
        cols = int(math.sqrt(n_images)) + 1
        rows = int(math.ceil(n_images / cols))
        plt.figure(figsize=(3.2*cols, 3.2*rows))
        
        for i in range(n_images):
            img = self._denormalize(mis_imgs[i], self.mean, self.std).permute(1, 2, 0).numpy()
            
            pred = mis_preds[i].item()
            true = mis_trues[i].item()
            
            ax = plt.subplot(cols, rows, i + 1)
            if img.shape[2] == 1:
                ax.imshow(img[..., 0], cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(f"Pred: {class_names[pred]} / True: {class_names[true]}", fontsize=9)
            ax.axis('off')
            
        plt.tight_layout()
        plt.show() 
        plt.close()
        

# ---------------Classification Report ------------------------

    def result_classification_report(self, model, data_loader, target_names, device):
        """ 
        Args:
            model: 学習済みのmodel
            data_loader: resultに使うdata
            target_names: class_names
            device: GPUで学習する場合
        
        example: result_classification_report(model=vit, data_loader=val_loader, target_names=class_names, device)
        """
        
        model.eval()  # 評価モード
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X, y in tqdm(data_loader, desc='Validation', total=len(data_loader), leave=False):
                X = X.to(device)
                y = y.to(device)
                logits = model(X)
                preds = logits.argmax(dim=1) 
        
                all_preds.append(preds.detach().cpu())
                all_targets.append(y.detach().cpu())
        
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()
        
        if target_names:
            target_names = target_names
        else:
            target_names = None  
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))