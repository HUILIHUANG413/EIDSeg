#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate fine-tuned segmentation models on the EIDSeg dataset.

- Supports HF models (Mask2Former / OneFormer / SegFormer / BEiT / EoMT)
- Supports external DeepLabV3+ repo model (VainF/DeepLabV3Plus-Pytorch)
- Outputs: mIoU (foreground only), pixAcc (foreground only), FWIoU (foreground only),
  per-class IoU/F1, FLOPs, Params, and a Confusion Matrix PDF (+ CSVs).

Assumptions:
- You have: datasets.py (UniversalSegmentationDataset, DeepLabV3PlusDataset)
- You have: models.py (load_model_and_processor, process_outputs_for_semantic)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image  # noqa: F401  (kept for PIL.Image compatibility in datasets)
from tqdm.auto import tqdm
from fvcore.nn import FlopCountAnalysis  # type: ignore

# Local modules you shared
from data import UniversalSegmentationDataset, DeepLabV3PlusDataset
from models import load_model_and_processor, process_outputs_for_semantic

# ---- Matplotlib headless for servers ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cv2  # type: ignore  # rasterize polygons (used inside datasets)


# --------------------------------------------------------------------------------------
# Globals
# --------------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"★ Using device: {DEVICE}")
np.set_printoptions(suppress=True, linewidth=140)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def _fast_hist(pred: np.ndarray, gt: np.ndarray, K: int) -> np.ndarray:
    k = (gt >= 0) & (gt < K)
    return np.bincount(K * gt[k].astype(int) + pred[k].astype(int),
                       minlength=K**2).reshape(K, K)


def compute_metrics(
    preds: List[np.ndarray],
    gts: List[np.ndarray],
    K: int,
    foreground_K: int = 5
):
    """Return (mIoU_foreground, pixAcc_foreground, FWIoU_foreground, iou_all, f1_all, hist)."""
    hist = np.zeros((K, K), dtype=np.float64)
    for p, g in zip(preds, gts):
        hist += _fast_hist(p.flatten(), g.flatten(), K)

    # all-class IoU (includes void in last index)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-6)

    # mIoU foreground only
    miou = iou[:foreground_K].mean()

    # pixel accuracy (foreground only)
    pred_flat = np.concatenate([p.flatten() for p in preds])
    gt_flat   = np.concatenate([g.flatten() for g in gts])
    fg_mask   = (gt_flat >= 0) & (gt_flat < foreground_K)
    pixel_acc = (pred_flat[fg_mask] == gt_flat[fg_mask]).mean()

    # frequency-weighted IoU (foreground only)
    freq = hist[:foreground_K].sum(1) / (hist[:foreground_K].sum() + 1e-6)
    fwiou = (freq[freq > 0] * iou[:foreground_K][freq > 0]).sum()

    precision = np.diag(hist) / (hist.sum(0) + 1e-6)
    recall    = np.diag(hist) / (hist.sum(1) + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return float(miou), float(pixel_acc), float(fwiou), iou, f1, hist


# --------------------------------------------------------------------------------------
# Viz helpers
# --------------------------------------------------------------------------------------
def plot_and_save_confmat(
    hist: np.ndarray,
    class_names: List[str],
    out_pdf: Path,
    normalize: bool = True,
    exclude_last: bool = True,
    fontsize: int = 12,
):
    cm = hist.copy()
    labels = class_names[:]
    if exclude_last:
        cm = cm[:-1, :-1]
        labels = labels[:-1]

    counts = cm
    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True) + 1e-12
        norm = counts / row_sums
    else:
        norm = counts

    fig, ax = plt.subplots(figsize=(1.1 * len(labels), 1.0 * len(labels)))
    im = ax.imshow(counts, aspect="auto")  # color = counts

    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=fontsize)
    ax.set_xlabel("Predicted", fontsize=fontsize + 1)
    ax.set_ylabel("Ground Truth", fontsize=fontsize + 1)
    ax.set_title("Confusion Matrix", fontsize=fontsize + 2)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = norm[i, j]
            cnt = int(counts[i, j])
            txt = f"{val*100:.1f}%\n({cnt})" if normalize else f"{cnt}"
            ax.text(j, i, txt, va="center", ha="center", fontsize=fontsize - 1)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    base = out_pdf.with_suffix("")
    np.savetxt(f"{base}_counts.csv", counts, fmt="%.0f", delimiter=",")
    np.savetxt(f"{base}_rownorm.csv", norm,   fmt="%.6f", delimiter=",")


# --------------------------------------------------------------------------------------
# FLOPs / params
# --------------------------------------------------------------------------------------
def model_stats(model: torch.nn.Module, example: torch.Tensor):
    try:
        flops = FlopCountAnalysis(model, (example,)).total()
    except Exception:
        flops = None
    params = sum(p.numel() for p in model.parameters())
    return flops, params


# --------------------------------------------------------------------------------------
# Evaluation core
# --------------------------------------------------------------------------------------
CLASS_NAMES = ["UD_Building", "D_Building", "Debris", "UD_Road", "D_Road", "void"]
NUM_CLASSES = 6
FOREGROUND_K = 5


def default_input_size(model_type: str, model_name: str) -> Tuple[int, int]:
    ln = model_name.lower()
    if "eomt" in ln:
        return (1024, 1024)
    if "beit" in ln:
        return (640, 640)
    # Keep 512x512 for others by default
    return (512, 512)


def safe_load_checkpoint_for_finetune(model: torch.nn.Module, ckpt_path: Path):
    """
    Load a fine-tuned checkpoint with tolerant key handling.
    """
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        # Try common wrappers
        for k in ["model_state_dict", "state_dict", "model_state", "net"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"✓ Loaded checkpoint: {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")


def make_loader_for_model(
    model_type: str,
    processor,
    images_dir: Path,
    annotation_xml: Path,
    image_size: Tuple[int, int],
    batch_size: int,
) -> DataLoader:
    if model_type == "deeplabv3plus":
        ds = DeepLabV3PlusDataset(
            annotation_xml=str(annotation_xml),
            data_dir=str(images_dir.parent),  # expects <data_dir>/default/<images>
            image_size=image_size,
            augment=False,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # HF family:
    ds = UniversalSegmentationDataset(
        annotation_xml=str(annotation_xml),
        data_dir=str(images_dir.parent),     # expects <data_dir>/default/<images>
        image_processor=processor,
        model_type=model_type,
        image_size=image_size,
        augment=False,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def evaluate_one_model(
    hub_or_tag: str,
    short_name: str,
    images_dir: Path,
    annotation_xml: Path,
    finetuned_path: Path | None,
    batch_size: int,
    deeplab_backbone: str,
    deeplab_os: int,
) -> Dict[str, float | str | None]:
    # Build model + processor
    img_size = default_input_size("", hub_or_tag)
    model, processor, model_type = load_model_and_processor(
        model_name=hub_or_tag,
        num_classes=NUM_CLASSES,
        image_size=img_size,
        deeplab_backbone=deeplab_backbone,
        deeplab_os=deeplab_os,
        deeplab_cityscapes_ckpt=None,  # not needed for evaluation here
    )
    model.to(DEVICE).eval()

    # Load fine-tuned weights if provided
    if finetuned_path is not None:
        safe_load_checkpoint_for_finetune(model, finetuned_path)

    # Dataloader
    loader = make_loader_for_model(
        model_type=model_type,
        processor=processor,
        images_dir=images_dir,
        annotation_xml=annotation_xml,
        image_size=img_size,
        batch_size=batch_size,
    )

    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []

    inner = tqdm(loader, desc=short_name, leave=False)
    with torch.no_grad():
        for batch in inner:
            if model_type == "deeplabv3plus":
                pv = batch["image"].to(DEVICE, non_blocking=True)  # [B,3,H,W]
                gt = batch["mask"].cpu().numpy()                   # [B,H,W] int
                out = model(pv)                                    # repo returns logits tensor
                # Unify & upsample to gt size
                logits = out if isinstance(out, torch.Tensor) else out.logits
                logits = torch.nn.functional.interpolate(
                    logits, size=gt.shape[-2:], mode="bilinear", align_corners=False
                )
                pred = logits.argmax(1).cpu().numpy()

            else:
                pv = batch["pixel_values"].to(DEVICE, non_blocking=True)
                gt = batch["pixel_mask"].cpu().numpy()

                if model_type == "oneformer":
                    ti = batch["task_inputs"].to(DEVICE, non_blocking=True)
                    outputs = model(pixel_values=pv, task_inputs=ti)
                else:
                    outputs = model(pixel_values=pv)

                # Unified conversion
                _, pred_t = process_outputs_for_semantic(
                    outputs=outputs,
                    target_size=gt.shape[-2:],   # (H,W)
                    model_type=model_type
                )
                pred = pred_t.cpu().numpy()

            preds.extend(list(pred))
            gts.extend(list(gt))

    # Metrics
    miou, pacc, fwiou, iou_vec, f1_vec, hist = compute_metrics(
        preds, gts, K=NUM_CLASSES, foreground_K=FOREGROUND_K
    )

    # Confusion matrix PDF per model
    return {
        "short_name": short_name,
        "model_type": model_type,
        "img_h": img_size[0],
        "img_w": img_size[1],
        "mIoU": round(miou, 4),
        "pixAcc": round(pacc, 4),
        "FWIoU": round(fwiou, 4),
        "hist": hist,
        "iou_vec": iou_vec,
        "f1_vec": f1_vec,
        "params": sum(p.numel() for p in model.parameters()) / 1e6,
        "flops": None if (lambda ex: False)(0) else _try_flops(model, img_size),
    }


def _try_flops(model: torch.nn.Module, img_size: Tuple[int, int]):
    try:
        dummy = torch.randn(1, 3, img_size[0], img_size[1], device=DEVICE)
        flops, _ = model_stats(model, dummy)
        return None if flops is None else flops / 1e9
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser("Evaluate segmentation models on EIDSeg")

    # Data
    ap.add_argument("--images", type=Path,
                    default=Path("../data/EIDSeg_Final_updated/test/images/default"),
                    help="Path to the *default* image folder (…/images/default)")
    ap.add_argument("--annotation", type=Path,
                    default=Path("../data/EIDSeg_Final_updated/test/test.xml"),
                    help="CVAT XML annotation file")

    # Output
    ap.add_argument("--csv_out", type=Path,
                    default=Path("evaluation_updated/test/results.csv"),
                    help="Where to save the summary CSV")
    ap.add_argument("--confmat_dir", type=Path,
                    default=Path("evaluation_updated/confmats"),
                    help="Directory to save confusion matrices")

    # Evaluate
    ap.add_argument("--batch_size", type=int, default=1)

    # Models to evaluate (hub_or_tag -> short name)
    ap.add_argument(
        "--models",
        nargs="+",
        default=[
            # Examples (uncomment to add HF baselines)
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024:segformer-b5",
            
            #"microsoft/beit-base-finetuned-ade-640-640: beit-base",
            #"microsoft/beit-large-finetuned-ade-640-640: beit-large",

            #"shi-labs/oneformer_cityscapes_swin_large: oneformer-large",

            #"tue-mps/cityscapes_semantic_eomt_large_1024: eomt-large",
            
            # "facebook/mask2former-swin-small-cityscapes-semantic: mask2former-swin-small",
            # "facebook/mask2former-swin-large-cityscapes-semantic: mask2former-swin-large",

            # Your local DeepLabV3+ tag for the external repo path:
            #"deeplabv3plus:deeplabv3plus",
        ],
        help=(
            "List of models as 'hub_or_tag:short_name'. "
            "Use 'deeplabv3plus:...' to load the external repo model."
        ),
    )
    

    # Fine-tuned checkpoints (short_name->path)
    ap.add_argument(
        "--ckpt",
        nargs="*",
        default=[
            # Example:
            "segformer-b5:../runs/segformer/Plan5includebackground_Final_512_512_b5_from_50/weights/best_model.pth",
            # "mask2former-s:/path/to/your_mask2former_best_model.pth",
            # Default for your DeepLabV3+ (you can override):
            #"deeplabv3plus:../DeepLabV3Plus-Pytorch/runs/deeplabv3plus/Plan5includebackground_Final_512_512/weights/best_model.pth",
        ],
        help="Optional list of 'short_name:/path/to/ckpt.pth'",
    )

    # DeepLabV3+ knobs
    ap.add_argument("--deeplab_backbone", type=str, default="resnet101",
                    choices=["resnet50", "resnet101", "xception", "mobilenet", "r50", "r101", "xcep", "mbv2"],
                    help="Backbone used in the external DeepLabV3+ repo")
    ap.add_argument("--deeplab_os", type=int, default=16, choices=[8, 16],
                    help="DeepLabV3+ output stride")

    return ap.parse_args()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    args = parse_args()

    # Parse MODEL_CATALOG
    model_catalog: Dict[str, str] = {}
    for spec in args.models:
        if ":" not in spec:
            raise ValueError(f"Model spec must be 'hub_or_tag:short_name', got: {spec}")
        hub, short = spec.split(":", 1)
        model_catalog[hub] = short

    # Parse checkpoints map
    finetuned_ckpts: Dict[str, Path] = {}
    for spec in (args.ckpt or []):
        if ":" not in spec:
            continue
        short, path = spec.split(":", 1)
        p = Path(path)
        if p.exists():
            finetuned_ckpts[short] = p
        else:
            print(f"⚠️  Checkpoint not found for {short}: {p} (skipping)")

    rows: List[Dict[str, float | str]] = []

    for hub_or_tag, short in tqdm(model_catalog.items(), desc="Evaluating", unit="model"):
        ckpt_path = finetuned_ckpts.get(short)

        result = evaluate_one_model(
            hub_or_tag=hub_or_tag,
            short_name=short,
            images_dir=args.images,
            annotation_xml=args.annotation,
            finetuned_path=ckpt_path,
            batch_size=args.batch_size,
            deeplab_backbone=args.deeplab_backbone,
            deeplab_os=args.deeplab_os,
        )

        # Save confusion matrix
        hist = result.pop("hist")
        iou_vec = result.pop("iou_vec")
        f1_vec = result.pop("f1_vec")

        plot_and_save_confmat(
            hist=hist,
            class_names=CLASS_NAMES,
            out_pdf=args.confmat_dir / f"confmat_{short}.pdf",
            normalize=True,
            exclude_last=True,
            fontsize=14,
        )

        # Flatten per-class IoU/F1
        for cname, val in zip(CLASS_NAMES, iou_vec):
            result[f"IoU_{cname}"] = round(float(val), 4)
        for cname, val in zip(CLASS_NAMES, f1_vec):
            result[f"F1_{cname}"] = round(float(val), 4)

        # Params / FLOPs pretty
        params_m = result.pop("params", None)
        flops_g  = result.pop("flops", None)
        if params_m is not None:
            result["Params(M)"] = round(float(params_m), 2)
        if flops_g is not None:
            result["FLOPs(G)"] = round(float(flops_g), 2)

        rows.append({
            "model": short,
            **{k: v for k, v in result.items() if not isinstance(v, (np.ndarray,))}
        })

    # Make CSV
    import pandas as pd  # lazy import
    df = pd.DataFrame(rows).set_index("model").sort_index()
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv_out)
    print("\nFinal results:\n", df)
    print(f"\n✓ Results saved to {args.csv_out}")


if __name__ == "__main__":
    main()
