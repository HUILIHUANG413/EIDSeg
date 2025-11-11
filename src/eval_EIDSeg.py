
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate fine-tuned segmentation models on the EIDSeg dataset.
Outputs mIoU, pixel accuracy, FWIoU, per-class IoU/F1, FLOPs and parameter count.
Excludes void class from mIoU, pixAcc, and FWIoU calculations.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import cv2                               # type: ignore
import numpy as np                       # type: ignore
import pandas as pd                      # type: ignore
import torch                             # type: ignore
from albumentations import Compose, Resize  # type: ignore
from fvcore.nn import FlopCountAnalysis  # type: ignore
from PIL import Image                    # type: ignore
from torch.utils.data import DataLoader
from tqdm.auto import tqdm               # notebook- & script-friendly
from transformers import (
    AutoImageProcessor,
    AutoModelForSemanticSegmentation,
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
    OneFormerProcessor,
    OneFormerForUniversalSegmentation,
    EomtForUniversalSegmentation,
)


import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"★ Using device: {DEVICE}")
import os

os.environ["USE_PYTORCH_KERNEL_CACHE"] = "0"
# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class UniversalSegmentationDataset_Plan5(torch.utils.data.Dataset):
    """Dataset reader that turns CVAT XML polygon annotations into semantic masks."""

    CLASS_MAP = {
        "UD_Building": 0,
        "D_Building" : 1,
        "Debris"     : 2,
        "UD_Road"    : 3,
        "D_Road"     : 4,
        "Undesignated": 5,
        "Background" : 5,
    }
    CLASS_NAMES = [
        "UD_Building", "D_Building", "Debris",
        "UD_Road", "D_Road", "void",
    ]
    NUM_CLASSES: int = 6
    FOREGROUND_CLASSES: int = 5  # Number of foreground classes (excluding void)

    def __init__(
        self,
        annotation_xml: str | Path,
        image_dir: str | Path,
        image_processor,
        image_size: Tuple[int, int] = (512, 512),
        model_type: str = "mask2former",
    ) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.image_processor = image_processor
        self.size = image_size
        self.model_type = model_type

        # collect image files
        self.filenames = sorted(
            p for p in self.image_dir.glob("**/*")
            if p.suffix.lower() in {".jpg", ".png"}
        )
        if not self.filenames:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        # parse XML once
        self.polygons = self._parse_xml(annotation_xml)

        # resize transform
        self.transform = Compose([Resize(height=self.size[0], width=self.size[1])])

    # ---------------- XML parsing ----------------
    @staticmethod
    def _parse_xml(xml_path: str | Path) -> Dict[str, List[Tuple[List[Tuple[float, float]], str]]]:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        poly_dict: Dict[str, List[Tuple[List[Tuple[float, float]], str]]] = {}
        for img in root.findall(".//image"):
            name = img.get("name")
            if name is None:
                continue
            for poly in img.findall(".//polygon"):
                label = poly.get("label") or "Background"
                pts = [
                    tuple(map(float, pt.split(",")))
                    for pt in (poly.get("points") or "").split(";")
                    if pt
                ]
                poly_dict.setdefault(name, []).append((pts, label))
        return poly_dict

    # -------------- rasterisation helpers --------------
    @staticmethod
    def _fill_poly(mask: np.ndarray, polygon, class_id: int):
        pts = np.round(np.array(polygon)).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], int(class_id))
        return mask

    def _make_mask(self, img_name: str, hw: Tuple[int, int]) -> np.ndarray:
        mask = np.full(hw[::-1], self.CLASS_MAP["Background"], dtype=np.uint8)
        for poly, lab in self.polygons.get(img_name, []):
            cid = self.CLASS_MAP.get(lab)
            if cid is not None:
                mask = self._fill_poly(mask, poly, cid)
        return mask

    # -------------- dataloader hooks --------------
    def __len__(self):  # noqa: D401
        return len(self.filenames)

    def __getitem__(self, idx: int):
        img_path = self.filenames[idx]
        img = Image.open(img_path).convert("RGB")
        mask = self._make_mask(img_path.name, img.size)

        aug = self.transform(image=np.asarray(img), mask=mask)
        img_aug, mask_aug = Image.fromarray(aug["image"]), aug["mask"]

        # OneFormer needs task_inputs = ["semantic"]
        if self.model_type == "oneformer":
            enc = self.image_processor(
                images=[img_aug],
                task_inputs=["semantic"],
                return_tensors="pt",
            )
            task_inputs = enc["task_inputs"].squeeze(0)
        else:
            enc = self.image_processor(images=img_aug, return_tensors="pt")
            task_inputs = None

        pixel_values = enc["pixel_values"].squeeze(0)
        mask_rs = cv2.resize(
            mask_aug,
            (pixel_values.shape[-1], pixel_values.shape[-2]),
            interpolation=cv2.INTER_NEAREST,
        )

        sample = {
            "pixel_values": pixel_values,
            "pixel_mask": torch.from_numpy(mask_rs).long(),
        }
        if task_inputs is not None:
            sample["task_inputs"] = task_inputs
        return sample


# --------------------------------------------------------------------------------------
# Metrics helpers
# --------------------------------------------------------------------------------------
def _fast_hist(pred: np.ndarray, gt: np.ndarray, K: int) -> np.ndarray:
    k = (gt >= 0) & (gt < K)
    return np.bincount(K * gt[k].astype(int) + pred[k].astype(int),
                       minlength=K ** 2).reshape(K, K)


def compute_metrics(preds, gts, K: int, foreground_K: int = 5):
    hist = np.zeros((K, K), dtype=np.float64)
    for p, g in zip(preds, gts):
        hist += _fast_hist(p.flatten(), g.flatten(), K)

    # Compute IoU for all classes (including void)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-6)
    
    # Compute mIoU only for foreground classes (excluding void, index 5)
    foreground_iou = iou[:foreground_K]
    miou = foreground_iou.mean()

    # Compute pixel accuracy only for foreground classes
    foreground_mask = np.isin(gts, range(foreground_K)).flatten()
    correct = (np.array(preds).flatten()[foreground_mask] == np.array(gts).flatten()[foreground_mask]).sum()
    total = foreground_mask.sum()
    pixel_acc = correct / (total + 1e-6)

    # Compute FWIoU only for foreground classes
    freq = hist[:foreground_K].sum(1) / (hist[:foreground_K].sum() + 1e-6)
    fwiou = (freq[freq > 0] * iou[:foreground_K][freq > 0]).sum()

    # Compute precision, recall, F1 for all classes
    precision = np.diag(hist) / (hist.sum(0) + 1e-6)
    recall = np.diag(hist) / (hist.sum(1) + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return miou, pixel_acc, fwiou, iou, f1,hist


# --------------------------------------------------------------------------------------
# Model utilities
# --------------------------------------------------------------------------------------
def resolution_for(model_name: str) -> Tuple[int, int]:
    name = model_name.lower()
    if "beit" in name:
        return 640, 640
    if "eomt" in name:
        return 1024, 1024
    return 512, 512


def load_model_and_processor(model_name: str, num_classes: int, image_size: Tuple[int, int]):
    size_cfg = {"height": image_size[0], "width": image_size[1]}

    if "segformer" in model_name.lower():
        proc = SegformerImageProcessor.from_pretrained(model_name, size=size_cfg)
        mdl = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        mdl_type = "segformer"

    elif "oneformer" in model_name.lower():
        proc = OneFormerProcessor.from_pretrained(model_name, size=size_cfg)
        mdl = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        in_feat = mdl.model.transformer_module.decoder.class_embed.in_features
        mdl.model.transformer_module.decoder.class_embed = torch.nn.Linear(in_feat, num_classes)
        mdl.config.num_labels = num_classes
        mdl_type = "oneformer"

    elif "eomt" in model_name.lower():
        proc = AutoImageProcessor.from_pretrained(
            model_name,
            crop_size=size_cfg,
            size={"shortest_edge": image_size[0], "longest_edge": image_size[1]},
        )
        mdl = EomtForUniversalSegmentation.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        mdl_type = "eomt"

    elif "mask2former" in model_name.lower():
        proc = Mask2FormerImageProcessor.from_pretrained(model_name, size=size_cfg)
        mdl = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name, ignore_mismatched_sizes=True
        )
        mdl.class_predictor = torch.nn.Linear(mdl.class_predictor.in_features, num_classes)
        mdl_type = "mask2former"

    elif "beit" in model_name.lower() or "deeplab" in model_name.lower():
        proc = AutoImageProcessor.from_pretrained(model_name, size=size_cfg)
        mdl = AutoModelForSemanticSegmentation.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        mdl_type = "beit" if "beit" in model_name.lower() else "deeplab"

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    mdl.to(DEVICE).eval()
    return mdl, proc, mdl_type


def model_stats(model: torch.nn.Module, example: torch.Tensor):
    try:
        flops = FlopCountAnalysis(model, (example,)).total()
    except Exception:
        flops = None
    params = sum(p.numel() for p in model.parameters())
    return flops, params


def safe_load_checkpoint(path: str | Path, model: torch.nn.Module):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    print(f"✓ Loaded checkpoint from {path}")


# --------------------------------------------------------------------------------------
# Evaluation loop
# --------------------------------------------------------------------------------------

def plot_and_save_confmat(
    hist: np.ndarray,
    class_names: List[str],
    out_pdf: Path,
    normalize: bool = True,
    exclude_last: bool = True,
    fontsize: int = 9,
):
    """
    Save a confusion matrix as a vector-quality PDF.
    - hist: KxK counts (rows = GT, cols = Pred)
    - class_names: names for all K classes (must match hist)
    - exclude_last: drop the last class (void) from the report
    - normalize: show row-normalized values as annotations; color map uses counts
    """
    cm = hist.copy()
    labels = class_names[:]

    if exclude_last:
        cm = cm[:-1, :-1]
        labels = labels[:-1]

    # counts for color scaling, row-normalized for annotations (optional)
    counts = cm
    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True) + 1e-12
        norm = counts / row_sums
    else:
        norm = counts

    # figure
    fig, ax = plt.subplots(figsize=(1.1*len(labels), 1.0*len(labels)))  # scales with classes
    im = ax.imshow(counts, aspect="auto")  # default colormap keeps it simple

    # ticks/labels
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=fontsize)
    ax.set_xlabel("Predicted", fontsize=fontsize+1)
    ax.set_ylabel("Ground Truth", fontsize=fontsize+1)
    ax.set_title("Confusion Matrix", fontsize=fontsize+2)

    # cell annotations (normalized values, with counts in parentheses)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = norm[i, j]
            cnt = int(counts[i, j])
            txt = f"{val*100:.1f}%\n({cnt})" if normalize else f"{cnt}"
            ax.text(j, i, txt, va="center", ha="center", fontsize=fontsize-1)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")  # PDF is vector—hi-res by nature
    plt.close(fig)

    # also save raw and normalized CSVs alongside
    base = out_pdf.with_suffix("")
    np.savetxt(f"{base}_counts.csv", counts, fmt="%.0f", delimiter=",")
    np.savetxt(f"{base}_rownorm.csv", norm,   fmt="%.6f", delimiter=",")

    
def evaluate_models(
    image_dir: Path,
    annotation_xml: Path,
    model_catalog: Dict[str, str],
    finetuned_ckpts: Dict[str, str] | None = None,
    batch_size: int = 1,
):
    rows: List[Dict[str, float | str | None]] = []

    outer = tqdm(model_catalog.items(), desc="Evaluating", unit="model")
    for hub_name, short_name in outer:
        img_size = resolution_for(hub_name)
        outer.set_postfix(model=short_name, size=f"{img_size[0]}×{img_size[1]}")

        model, processor, model_type = load_model_and_processor(
            hub_name, UniversalSegmentationDataset_Plan5.NUM_CLASSES, img_size
        )

        ckpt = (finetuned_ckpts or {}).get(short_name)
        if ckpt:
            safe_load_checkpoint(ckpt, model)

        dataset = UniversalSegmentationDataset_Plan5(
            annotation_xml, image_dir, processor, img_size, model_type=model_type
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds, gts = [], []
        inner = tqdm(loader, desc=short_name, leave=False, position=1)
        with torch.no_grad():
            for batch in inner:
                pv = batch["pixel_values"].to(DEVICE, non_blocking=True)
                gt = batch["pixel_mask"].cpu().numpy()

                if model_type == "oneformer":
                    ti = batch["task_inputs"].to(DEVICE, non_blocking=True)
                    out = model(pixel_values=pv, task_inputs=ti)
                else:
                    out = model(pixel_values=pv)

                # SegFormer / BEiT / DeepLab
                if hasattr(out, "logits"):
                    logits = torch.nn.functional.interpolate(
                        out.logits,
                        size=gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred = logits.argmax(1).cpu().numpy()
                # Query-based decoders
                else:
                    cls = torch.softmax(out.class_queries_logits, dim=-1)
                    msk = torch.sigmoid(
                        torch.nn.functional.interpolate(
                            out.masks_queries_logits,
                            size=gt.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                    sem = torch.einsum("bqc,bqhw->bchw", cls, msk)
                    pred = sem.argmax(1).cpu().numpy()

                preds.extend(pred)
                gts.extend(gt)

        miou, pacc, fwiou, iou_vec, f1_vec,hist = compute_metrics(
            preds, gts, 
            UniversalSegmentationDataset_Plan5.NUM_CLASSES,
            UniversalSegmentationDataset_Plan5.FOREGROUND_CLASSES
        )

        cn = UniversalSegmentationDataset_Plan5.CLASS_NAMES

        confmat_pdf = (args.confmat_dir if "args" in globals() else Path("evaluation_updated/confmats")) / f"confmat_{short_name}.pdf"
        plot_and_save_confmat(
            hist=hist,
            class_names=cn,
            out_pdf=confmat_pdf,
            normalize=True,
            exclude_last=True,   # drop "void" in the visual
            fontsize=15,
        )

        
        class_iou = dict(zip([f"IoU_{n}" for n in cn], np.round(iou_vec, 3)))
        class_f1  = dict(zip([f"F1_{n}"  for n in cn], np.round(f1_vec, 3)))

        dummy = torch.randn(1, 3, *img_size).to(DEVICE)
        flops, params = model_stats(model, dummy)

        rows.append({
            "model": short_name,
            "mIoU": round(float(miou), 3),
            "pixAcc": round(float(pacc), 3),
            "FWIoU": round(float(fwiou), 3),
            "FLOPs(G)": None if flops is None else round(flops / 1e9, 2),
            "Params(M)": round(params / 1e6, 2),
            **class_iou,
            **class_f1,
        })
        print(rows)

    return pd.DataFrame(rows).set_index("model")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser("Evaluate segmentation models on EIDSeg")
    ap.add_argument(
        "--images", type=Path,
        default="data/test/images",
        help="Folder with evaluation images",
    )
    ap.add_argument(
        "--annotation", type=Path,
        default="data/test/test.xml",
        help="CVAT XML annotation file",
    )
    ap.add_argument(
        "--csv_out", type=Path,
        default="evaluation_updated/test/beitb.csv",
        help="Where to save CSV results",
    )
    ap.add_argument("--batch_size", type=int, default=1)
    return ap.parse_args()

    # ap.add_argument(
    # "--confmat_dir", type=Path,
    # default=Path("evaluation_updated/confmats"),
    # help="Directory to save confusion-matrix PDFs/CSVs per model",)



# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    args = parse_args()


    MODEL_CATALOG:Dict[str, str] = {
       #"nvidia/segformer-b5-finetuned-cityscapes-1024-1024": "segformer-b5",
       "microsoft/beit-base-finetuned-ade-640-640": "beit-base",
        #"microsoft/beit-large-finetuned-ade-640-640": "beit-large",
       # "facebook/deeplabv3-resnet101": "deeplabv3+",
        #"shi-labs/oneformer_cityscapes_swin_large": "oneformer-large",
        #"tue-mps/cityscapes_semantic_eomt_large_1024": "eomt-large",
        # "facebook/mask2former-swin-small-cityscapes-semantic": "mask2former-swin-small",
       # "facebook/mask2former-swin-large-cityscapes-semantic": "mask2former-swin-large",
    }

    # Optional: fine‑tuned checkpoint paths (short name -> .pth)
    FINETUNED_CKPTS: Dict[str, str] = {
        
       #"segformer-b5": "runs/segformer/Plan5includebackground_Final_512_512_b5_from_50/weights/best_model.pth",
        "beit-base":         "runs/beit/Plan5includebackground_Final_640_640_b5_base/weights/best_model.pth",
        #"beit-large":        "runs/beit/Plan5includebackground_Final_640_640_large_from_epoch17/weights/best_model.pth",
        #"oneformer-large":   "runs/oneformer/Plan5includebackground_Final_512_512_Large_From_Epoch10_2025-07-02_01-36-44/weights/best_model.pth",
       # "eomt-large":        "runs/eomt/Plan5includebackground_Final_1024_1024_emot_large_from_epoch19/weights/best_model.pth",
        #"mask2former-swin-small": "runs/Plan5includebackground_Final_512_512_Small_From_epoch58_2025-07-01_23-19-39/weights/best_model.pth",
        #"mask2former-swin-large": "runs/Plan5includebackground_Final_512_512_Large_From_epoch10_2025-07-01_21-41-16/weights/best_model.pth",

    }

    df = evaluate_models(
        image_dir=args.images,
        annotation_xml=args.annotation,
        model_catalog=MODEL_CATALOG,
        finetuned_ckpts=FINETUNED_CKPTS,
        batch_size=args.batch_size,
    )

    print("\nFinal results:\n", df, sep="")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_out)
    print(f"\n✓ Results saved to {args.csv_out}")


if __name__ == "__main__":
    main()
