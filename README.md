# Dog vs Cat — Image Classification (PyTorch + timm + Albumentations)

本リポジトリは、犬/猫の2クラス分類を題材に、**再現性のある学習パイプライン**を示すポートフォリオです。  
- フレームワーク: PyTorch, timm, Albumentations  
- 可視化: Confusion Matrix / Misclassified Samples / Grad-CAM  
- 設定管理: `configs/config.yaml`  
- 実行結果の保存: `runs/<timestamp>_<model>_bs<...>_lr<...>_ep<...>/`

---

## Table of Contents
- [Project Structure](#project-structure)
- [Environment](#environment)
- [Dataset](#dataset)
- [How to Run](#how-to-run)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Results (Samples)](#results-samples)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Project Structure