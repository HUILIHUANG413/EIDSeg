

# EIDSeg: A Pixel-Level Semantic Segmentation Dataset for Post-Earthquake Damage Assessment from Social Media Images

### Paper(Camera-Ready) | [Extended Version](https://arxiv.org/abs/2511.06456) | Dataset(under review)

## Overview of EIDSeg
The code in this repository implements the data preparation, model training, and evaluation protocols as described in paper *EIDSeg: A Pixel-Level Semantic Segmentation Dataset for Post-Earthquake Damage Assessment from Social Media Images* in AAAI-AISI 2026. In this repo, we unified training pipeline for the **EIDSeg** dataset (post-earthquake infrastructure damage segmentation) supporting:

* **Hugging Face** models: Mask2Former, OneFormer, SegFormer, BEiT, EoMT
* **DeepLabV3+** from the [VainF repo](https://github.com/VainF/DeepLabV3Plus-Pytorch))



## 📦 Repository Layout

```
.
├─ src/
│  ├─ datasets.py            # Dataset Function
│  ├─ models.py              # model factory (HF + DeepLabV3+)
│  ├─ eval_EIDSeg.py         # CLI evaluation entry
│  └─ train_EIDSeg.py        # CLI training entry point (all models)
├─ data/                     # your dataset lives here (see “Data Layout” below)
└─ DeepLabV3Plus-Pytorch/    # external repo (optional; required for DeepLabV3+)
```



## 🧰 Installation

Follow the steps below to set up the environment and run the EIDSeg experiments.

### 1. Clone the Repository

```bash
git clone https://github.com/HUILIHUANG413/EIDSeg.git
cd EIDSeg
```

### 2. Create a Conda Environment

It is recommended to use **Python 3.10+**.

```bash
conda create -n eidseg python=3.12 -y
conda activate eidseg
```

### 3. Install Dependencies

All required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```


Optional (for DeepLabV3+):

```bash
git clone https://github.com/VainF/DeepLabV3Plus-Pytorch.git
# (optional but convenient)
pip install -e DeepLabV3Plus-Pytorch
```

<!-- > If you’re on a cluster (e.g., PACE), make sure `$HOME/.cache/huggingface` has quota and proper permissions, or set `HF_HOME` to a writable path. -->


## 📁 Data Layout

The code expects **CVAT-style XML** annotations and images arranged like:

```
data/
└── train/
  │  ├─ train.xml
  │  └─ images/
  │     └─ default/
  │        ├─ 0001.jpg
  │        ├─ 0002.png
  │        └─ ...
  └─ val/
    ├─ val.xml
    └─ images/
        └─ default/
          ├─ 1001.jpg
          └─ ...
```

**Annotations** (CVAT XML):

```xml
<annotations>
  <image name="0001.jpg" ...>
    <polygon label="D_Building" points="x1,y1;x2,y2;..." />
    <polygon label="UD_Road"    points="..." />
    ...
  </image>
</annotations>
```

**Class mapping (6 classes):**

```
0: UD_Building
1: D_Building
2: Debris
3: UD_Road
4: D_Road
5: void (Background / Undesignated)
```



## ▶️ Quick Start

### A) Train a Hugging Face model (e.g., EoMT)

```bash
python train_EIDSeg.py \
  --train-xml   ../data/train/train.xml \
  --train-imgdir ../data/train/images \
  --val-xml     ../data/val/val.xml \
  --val-imgdir   ../data/val/images \
  --model-name   tue-mps/cityscapes_semantic_eomt_large_1024 \
  --image-size   1024 1024 \
  --epochs 50 --batch-size 1 --lr 1e-5 \
  --augment \
  --run-dir runs/eomt/Plan5_1024_1024
```

Supported HF models in paper:

- `"nvidia/segformer-b5-finetuned-cityscapes-1024-1024"`
- `"microsoft/beit-base-finetuned-ade-640-640"`
- `"microsoft/beit-large-finetuned-ade-640-640"`
- `"shi-labs/oneformer_cityscapes_swin_large"`
- `"tue-mps/cityscapes_semantic_eomt_large_1024"`
- `"facebook/mask2former-swin-small-cityscapes-semantic"`
- `"facebook/mask2former-swin-large-cityscapes-semantic"`


### B) Train **DeepLabV3+** (VainF repo, optional)

1. Make sure the external repo exists at `./DeepLabV3Plus-Pytorch/`.
2. (Optional) Download a Cityscapes checkpoint (e.g.,
   `best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar`) and note the path.

```bash
python train_EIDSeg.py \
  --train-xml   ../data/train/train.xml \
  --train-imgdir ./data/train/images \
  --val-xml     ../data/val/val.xml \
  --val-imgdir   ../data/val/images \
  --model-name   deeplabv3plus-resnet101 \
  --deeplab-backbone resnet101 \
  --deeplab-os 16 \
  --deeplab-cityscapes-ckpt DeepLabV3Plus-Pytorch/weights/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar \
  --image-size 512 512 \
  --epochs 150 --batch-size 2 --lr 1e-5 \
  --augment \
  --run-dir runs/deeplabv3plus/Plan5_512_512
```

<!-- > The trainer auto-detects DeepLab when `--model-name` starts with `deeplabv3plus-`. -->



## 🗂️ Outputs

Inside `--run-dir`:

```
run_dir/
├─ training_log.txt
├─ hyperparameters.json
├─ loss_plot.png
└─ weights/
   ├─ best_model.pth
   ├─ final_model.pth
   └─ checkpoint.pth     # contains epoch, optimizer/scheduler states, curves, best_miou
```
Here are clean, properly formatted **Markdown versions** of both tables for your README.

## Benchmark Results
Semantic Segmentation Benchmark of EIDSeg

| Model           | Backbone   | Pre-train  | Input  | mIoU (%) | FWIoU (%) | PA (%) | FLOPs (G) | Params (M) |
|:---------------:|:----------:|:----------:|:------:|:--------:|:---------:|:------:|:---------:|:----------:|
| DeepLabV3+       | ResNet-101 | Cityscapes | 512²   | 67.1     | 68.2      | 86.0   | 79.29     | 58.76      |
| SegFormer        | MiT-B5     | Cityscapes | 512²   | 74.4     | 75.2      | 86.9   | 110.16    | 84.60      |
| Mask2Former-S    | Swin-S     | Cityscapes | 512²   | 76.1     | 77.1      | 87.7   | 93.21     | 81.42      |
| Mask2Former-L    | Swin-L     | Cityscapes | 512²   | 77.4     | 78.4      | 88.7   | 250.54    | 215.45     |
| BEiT-B           | ViT-B      | ADE20K     | 640²   | 78.7     | 79.6      | 89.8   | 1823.53   | 441.09     |
| BEiT-L           | ViT-L      | ADE20K     | 640²   | 79.0     | 79.8      | 89.9   | 3182.73   | 311.62     |
| OneFormer        | Swin-L     | Cityscapes | 512²   | 79.8     | 80.2      | 89.8   | 1042.14   | 218.77     |
| **EoMT**         | ViT-L      | Cityscapes | 1024²  | **80.8** | **80.9**  | **90.3** | 1341.85 | 319.02     |



Class-wise IoU and mIoU (%) for each model on EIDSeg

| Model          | UD_Building | D_Building | Debris | UD_Road | D_Road | mIoU (%) |
|:--------------:|:-----------:|:-----------:|:------:|:-------:|:------:|:--------:|
| DeepLabV3+     | 34.5        | 65.4        | 77.3   | 75.7    | 73.7   | 67.1     |
| SegFormer      | 54.9        | 73.5        | 82.3   | 79.9    | 79.4   | 74.4     |
| Mask2Former-S  | 58.9        | 76.7        | 83.8   | 80.2    | 80.1   | 76.1     |
| Mask2Former-L  | 63.5        | 76.9        | 84.9   | 82.0    | 80.9   | 77.4     |
| BEiT-B         | 66.0        | 76.7        | **85.1** | 82.3  | 78.7   | 78.7     |
| BEiT-L         | 66.4        | 77.9        | **85.1** | 82.6  | 78.7   | 79.0     |
| OneFormer      | 68.7        | 79.7        | 85.0   | **84.1** | 79.9 | 79.8     |
| **EoMT**       | **70.1**    | **80.0**    | 84.6   | 82.0    | **87.3** | **80.8** |




## 🙌 Acknowledgments

The research described herein was supported in part by the School of Computing Instruction and the Elizabeth and Bill Higginbotham Professorship at Georgia Tech. This support is gratefully acknowledged. Support for the undergraduate students participating in the project was provided by the US National Science Foundation through the Geotechnical Engineering Program under Grant No. CMMI-1826118. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of NSF.

## License & Credit

* DeepLabV3+ implementation courtesy of **VainF**:
  [https://github.com/VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)
* Our Classification dataset **EID**: https://github.com/HUILIHUANG413/Earthquake-Infrastracture-Damage 



## Contact
Huili Huang - huilihuang1997@gmail.com; hhuang413@gatech.edu

Please ⭐ if you find it useful so that I find the motivation to keep improving this. Thanks




Here’s a clean, professional paragraph you can use in your README:



## Citation

If you find this work or the EIDSeg dataset useful in your research, please consider citing our paper. Your citation helps support and encourage future development of this project.

```
@article{huang2025eidseg,
  title   = {EIDSeg: Post-Earthquake Infrastructure Damage Segmentation Dataset},
  author  = {Huili Huang and Chengeng Liu and Danrong Zhang and Shail Patel and Anastasiya Masalava and Sagar Sadak and Parisa Babolhavaeji and Weihong Low and Max Mahdi Roozbahani and J.~David Frost},
  journal = {arXiv preprint arXiv:XXXXX},
  year    = {2025}
}
```

```
@article{huang2025enhancing_fidelity,
  title   = {Enhancing the Fidelity of Social Media Image Datasets in Earthquake Damage Assessment},
  author  = {Huili Huang and Chengeng Liu and Danrong Zhang and Shail Patel and Anastasiya Masalava and Sagar Sadak and Parisa Babolhavaeji and Weihong Low and Max Mahdi Roozbahani and J.David Frost},
  journal = {Earthquake Spectra},
  volume  = {41},
  number  = {3},
  pages   = {2616-2635},
  year    = {2025},
  doi     = {10.1177/87552930251335649},
  url     = {https://doi.org/10.1177/87552930251335649}
}
```
