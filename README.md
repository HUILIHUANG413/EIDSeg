

# EIDSeg: A Pixel-Level Semantic Segmentation Dataset for Post-Earthquake Damage Assessment from Social Media Images

### [Paper]() | [Extended Version]() | Dataset(under review)

## Overview of EIDSeg
The code in this repository implements the data preparation, model training, and evaluation protocols as described in paper *EIDSeg: A Pixel-Level Semantic Segmentation Dataset for Post-Earthquake Damage Assessment from Social Media Images* in AAAI-AISI 2026. In this repo, we unified training pipeline for the **EIDSeg** dataset (post-earthquake infrastructure damage segmentation) supporting:

* **Hugging Face** models: Mask2Former, OneFormer, SegFormer, BEiT, EoMT
* **DeepLabV3+** from the [VainF repo](https://github.com/VainF/DeepLabV3Plus-Pytorch))

---

## 📦 Repository Layout

```
.
├─ src/
│  ├─ datasets.py            # Dataset Function
│  ├─ models.py              # model factory (HF + DeepLabV3+), unified output processing
│  └─ train_EIDSeg.py        # CLI training entry point (all models)
├─ data/                     # your dataset lives here (see “Data Layout” below)
└─ DeepLabV3Plus-Pytorch/    # external repo (optional; required for DeepLabV3+)
```

---

## 🧰 Installation
(git to this repo)
(create enviroment)
(install by requirement.txt)

Here is a clean, professional “Installation” section you can directly paste into your README. I kept it concise and consistent with typical ML-paper repositories:

---

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
conda create -n eidseg python=3.10 -y
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

---

## 📁 Data Layout

The code expects **CVAT-style XML** annotations and images arranged like:

```
data/
└─ EIDSeg_Final_updated/
   ├─ train/
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

---

## ▶️ Quick Start

### A) Train a Hugging Face model (e.g., EoMT)

```bash
python -m src.train_EIDSeg \
  --train-xml   data/EIDSeg_Final_updated/train/train.xml \
  --train-imgdir data/EIDSeg_Final_updated/train/images \
  --val-xml     data/EIDSeg_Final_updated/val/val.xml \
  --val-imgdir   data/EIDSeg_Final_updated/val/images \
  --model-name   tue-mps/cityscapes_semantic_eomt_large_1024 \
  --image-size   1024 1024 \
  --epochs 50 --batch-size 1 --lr 1e-5 \
  --augment \
  --run-dir runs/eomt/Plan5_1024_1024
```

Supported HF models in paper:

* `facebook/mask2former-swin-small-cityscapes-semantic`
* `shi-labs/oneformer_cityscapes_swin_large`
* `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`
* `microsoft/beit-base-finetuned-ade-640-640`
* ...

### B) Train **DeepLabV3+** (VainF repo, optional)

1. Make sure the external repo exists at `./DeepLabV3Plus-Pytorch/`.
2. (Optional) Download a Cityscapes checkpoint (e.g.,
   `best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar`) and note the path.

```bash
python -m src.train_EIDSeg \
  --train-xml   data/EIDSeg_Final_updated/train/train.xml \
  --train-imgdir data/EIDSeg_Final_updated/train/images \
  --val-xml     data/EIDSeg_Final_updated/val/val.xml \
  --val-imgdir   data/EIDSeg_Final_updated/val/images \
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

---

## 🔧 Command-Line Arguments

**Common**

* `--train-xml, --train-imgdir`: CVAT XML and image root (expects `default/` inside)
* `--val-xml, --val-imgdir`: validation XML and images
* `--model-name`: HF model id or `deeplabv3plus-<backbone>`
* `--image-size H W`: target size (e.g., `1024 1024` for HF; `512 512` common for DeepLab)
* `--epochs, --batch-size, --lr`
* `--augment`: enable simple augmentation (flip, brightness/contrast)
* `--run-dir`: output directory (logs, weights, plots)
* `--resume`: resume from a previous `checkpoint.pth`
* `--num-workers`: dataloader workers

**DeepLabV3+ only**

* `--deeplab-backbone`: `resnet50|resnet101|xception|mobilenet`
* `--deeplab-os`: output stride, `8` or `16`
* `--deeplab-cityscapes-ckpt`: optional Cityscapes pretrain `.pth.tar`

---

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

---

## 🧠 How It Works (High Level)

* **Datasets**

  * `EIDSegDataset`: parses CVAT XML and lists images under `default/`.
  * `UniversalSegmentationDataset`: builds masks from polygons and uses a **HF image processor** (Mask2Former, OneFormer, SegFormer, BEiT, EoMT).
  * `DeepLabV3PlusDataset`: torchvision transforms + ImageNet normalization for DeepLab; masks resized to target size.

* **Models**

  * `models.py` creates models and processors:

    * HF models: adapts classifier to 6 classes.
    * DeepLabV3+: imports from external repo. Optionally loads Cityscapes checkpoint and drops incompatible classifier weights automatically.

* **Training**

  * Unified forward → `process_outputs_for_semantic` produces `(B,C,H,W)` logits and argmax predictions for **all** models.
  * Tracks **mIoU** (macro over six classes), saves **best** and **final** checkpoints, and logs to `training_log.txt`.



## 📜 License & Credit

* This training wrapper is yours to license as you wish.
* DeepLabV3+ implementation courtesy of **VainF**:
  [https://github.com/VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)
* Model weights and configs referenced from **Hugging Face Transformers**.

---

## 🙌 Acknowledgments

Thanks to contributors of EID/EIDSeg, and maintainers of the referenced open-source libraries.

---

If you want, I can also add:

* `requirements.txt` (exact versions you use on PACE),
* a minimal `inference.py` to export predicted masks for a folder,
* and a badges/figures section for your GitHub page.
