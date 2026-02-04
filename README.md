# Get Start

## 1.Install

```bash
conda create -n SFENet python=3.10 -y
conda activate SFENet
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmcv-full==1.7.2
pip install timm lmdb mmengine thop numpy==1.26.4 opencv-python==4.8.1.78
```

## 2.**Download Datasets**

Organize your dataset as follows:

```
datasets/
└── CrackMap/
    ├── train_img/
    ├── train_lab/
    ├── val_img/
    ├── val_lab/
    ├── test_img/
    └── test_lab/
```

## 3.Training

```bash
python train.py --dataset_path datasets/CrackMap
```

## 4.Testing

Place the pretrained weights in `checkpoint/`, then run:

```bash
python test.py
```
