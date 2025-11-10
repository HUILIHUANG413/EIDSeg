#!/usr/bin/env python3
# src/train_EIDSeg.py
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt



from data import UniversalSegmentationDataset, DeepLabV3PlusDataset
from models import load_model_and_processor, process_outputs_for_semantic

from data import UniversalSegmentationDataset as USD

# Perf knobs
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def get_args():
    p = argparse.ArgumentParser(description="Train EIDSeg (semantic segmentation)")

    # Data
    p.add_argument("--train-xml", required=True, type=str)
    p.add_argument("--train-imgdir", required=True, type=str)
    p.add_argument("--val-xml", required=True, type=str)
    p.add_argument("--val-imgdir", required=True, type=str)

    # Model selector
    p.add_argument("--model-name", required=True, type=str,
                   help=("HF id or 'deeplabv3plus-*' marker, e.g. "
                         "'deeplabv3plus-resnet101', 'facebook/mask2former-swin-small-cityscapes-semantic', "
                         "'tue-mps/cityscapes_semantic_eomt_large_1024'"))

    # Shared HP
    p.add_argument("--image-size", nargs=2, type=int, default=[1024, 1024], metavar=("H", "W"))
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--augment", action="store_true")

    # DeepLabV3+ specific (used only when --model-name startswith 'deeplabv3plus')
    p.add_argument("--deeplab-backbone", type=str, default="resnet101",
                   help="resnet50|resnet101|xception|mobilenet")
    p.add_argument("--deeplab-os", type=int, default=16, choices=[8, 16],
                   help="output stride for DeepLabV3+ (8 or 16)")
    p.add_argument("--deeplab-cityscapes-ckpt", type=str, default="",
                   help="Path to Cityscapes pretrain .pth.tar from VainF repo")

    # Runtime
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def plot_losses(train_losses, val_losses, run_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val', marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curves'); plt.grid(True); plt.legend()
    plt.savefig(run_dir / 'loss_plot.png', bbox_inches='tight'); plt.close()


def train_one_epoch(model, loader, device, criterion, optimizer, model_type: str):
    model.train()
    #from datasets import UniversalSegmentationDataset as USD
    num_classes = USD.NUM_CLASSES
    running_loss = 0.0
    total_inter = torch.zeros(num_classes, dtype=torch.float64, device="cpu")
    total_union = torch.zeros(num_classes, dtype=torch.float64, device="cpu")

    bar = tqdm(loader, desc="Train", leave=False)
    for batch in bar:
        optimizer.zero_grad()

        if model_type == "deeplabv3plus":
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            outputs = model(images)                    # logits
            # normalize to (B,C,H,W) at target size (already correct by dataset)
            target_size = masks.shape[1:]
            semantic_logits, preds = process_outputs_for_semantic(outputs, target_size, model_type)
            loss = criterion(semantic_logits, masks)
        else:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask   = batch["pixel_mask"].to(device)
            if model_type == "oneformer" and "task_inputs" in batch:
                outputs = model(pixel_values=pixel_values, task_inputs=batch["task_inputs"].to(device))
            elif "patch_start_indices" in batch:
                outputs = model(pixel_values=pixel_values, patch_start_indices=batch["patch_start_indices"].to(device))
            else:
                outputs = model(pixel_values=pixel_values)
            target_size = pixel_mask.shape[1:]
            semantic_logits, preds = process_outputs_for_semantic(outputs, target_size, model_type)
            loss = criterion(semantic_logits, pixel_mask)

        loss.backward(); optimizer.step()
        running_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")

        # IoU accumulators
        tgt = masks if model_type == "deeplabv3plus" else pixel_mask
        for cls in range(num_classes):
            inter = ((preds == cls) & (tgt == cls)).sum().item()
            union = ((preds == cls) | (tgt == cls)).sum().item()
            total_inter[cls] += inter; total_union[cls] += union

    avg_loss = running_loss / max(1, len(loader))
    iou_per_class = (total_inter / (total_union + 1e-6)).cpu().numpy()
    miou = float(np.mean(iou_per_class))
    #from data import UniversalSegmentationDataset as USD2
    iou_dict = dict(zip(USD.CLASS_NAMES, iou_per_class.tolist()))
    return avg_loss, miou, iou_dict


@torch.no_grad()
def evaluate(model, loader, device, criterion, model_type: str):
    model.eval()
    #from data import UniversalSegmentationDataset as USD
    num_classes = USD.NUM_CLASSES
    running_loss = 0.0
    total_inter = torch.zeros(num_classes, dtype=torch.float64, device="cpu")
    total_union = torch.zeros(num_classes, dtype=torch.float64, device="cpu")

    bar = tqdm(loader, desc="Val", leave=False)
    for batch in bar:
        if model_type == "deeplabv3plus":
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            outputs = model(images)
            target_size = masks.shape[1:]
            semantic_logits, preds = process_outputs_for_semantic(outputs, target_size, model_type)
            loss = criterion(semantic_logits, masks)
            tgt = masks
        else:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask   = batch["pixel_mask"].to(device)
            if model_type == "oneformer" and "task_inputs" in batch:
                outputs = model(pixel_values=pixel_values, task_inputs=batch["task_inputs"].to(device))
            elif "patch_start_indices" in batch:
                outputs = model(pixel_values=pixel_values, patch_start_indices=batch["patch_start_indices"].to(device))
            else:
                outputs = model(pixel_values=pixel_values)
            target_size = pixel_mask.shape[1:]
            semantic_logits, preds = process_outputs_for_semantic(outputs, target_size, model_type)
            loss = criterion(semantic_logits, pixel_mask)
            tgt = pixel_mask

        running_loss += loss.item()
        for cls in range(num_classes):
            inter = ((preds == cls) & (tgt == cls)).sum().item()
            union = ((preds == cls) | (tgt == cls)).sum().item()
            total_inter[cls] += inter; total_union[cls] += union

    avg_loss = running_loss / max(1, len(loader))
    iou_per_class = (total_inter / (total_union + 1e-6)).cpu().numpy()
    miou = float(np.mean(iou_per_class))
    #from data import UniversalSegmentationDataset as USD2
    iou_dict = dict(zip(USD.CLASS_NAMES, iou_per_class.tolist()))
    return avg_loss, miou, iou_dict


def main():
    args = get_args()

    run_dir = Path(args.run_dir); (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=run_dir / "training_log.txt",
                        level=logging.INFO,
                        format="%(asctime)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)

    
    hyperparams = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,
        "image_size": args.image_size,
        "model_name": args.model_name,
        "num_classes": USD.NUM_CLASSES,
        "augment": args.augment,
        "deeplab_backbone": args.deeplab_backbone,
        "deeplab_os": args.deeplab_os,
        "deeplab_cityscapes_ckpt": args.deeplab_cityscapes_ckpt
    }
    (run_dir / "hyperparameters.json").write_text(json.dumps(hyperparams, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | GPUs: {torch.cuda.device_count()}")

    # Build model (and processor for HF paths)
    model, processor, model_type = load_model_and_processor(
        args.model_name, USD.NUM_CLASSES, tuple(args.image_size),
        deeplab_backbone=args.deeplab_backbone,
        deeplab_os=args.deeplab_os,
        deeplab_cityscapes_ckpt=args.deeplab_cityscapes_ckpt
    )
    print(f"Model type: {model_type}")
    logger.info(f"Model type: {model_type}")

    model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    model = model.to(device)

    # Datasets / loaders
    if model_type == "deeplabv3plus":
        train_ds = DeepLabV3PlusDataset(args.train_xml, args.train_imgdir,
                                        image_size=tuple(args.image_size), augment=args.augment)
        val_ds   = DeepLabV3PlusDataset(args.val_xml, args.val_imgdir,
                                        image_size=tuple(args.image_size), augment=False)
    else:
        train_ds = UniversalSegmentationDataset(args.train_xml, args.train_imgdir, processor,
                                                model_type=model_type,
                                                image_size=tuple(args.image_size), augment=args.augment)
        val_ds   = UniversalSegmentationDataset(args.val_xml, args.val_imgdir, processor,
                                                model_type=model_type,
                                                image_size=tuple(args.image_size), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=(model_type=="deeplabv3plus"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=(model_type=="deeplabv3plus"))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4 if model_type=="deeplabv3plus" else 0.0)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, verbose=True)
    criterion = CrossEntropyLoss()

    best_miou, start_epoch = 0.0, 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    if args.resume and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt["model_state_dict"]
        (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(state, strict=False)
        if "optimizer_state_dict" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        best_miou = ckpt.get("best_miou", 0.0)

    patience, epochs_no_improve = 30, 0
    weights_dir = run_dir / "weights"

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        tr_loss, tr_miou, _ = train_one_epoch(model, train_loader, device, criterion, optimizer, model_type)
        vl_loss, vl_miou, _ = evaluate(model, val_loader, device, criterion, model_type)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        print(f"Train | loss={tr_loss:.4f} mIoU={tr_miou:.4f}")
        print(f"Val   | loss={vl_loss:.4f} mIoU={vl_miou:.4f}")
        logger.info(f"Epoch {epoch+1}/{args.epochs} | train_loss={tr_loss:.4f} train_mIoU={tr_miou:.4f} | val_loss={vl_loss:.4f} val_mIoU={vl_miou:.4f}")

        if vl_miou > best_miou:
            best_miou = vl_miou
            to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(to_save.state_dict(), weights_dir / "best_model.pth")
            print(f"✓ Saved best model (val mIoU={best_miou:.4f})")
            logger.info(f"Saved best model (val mIoU={best_miou:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping (no improvement for {patience} epochs).")
                break

        scheduler.step(vl_miou)
        plot_losses(train_losses, val_losses, run_dir)

        to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_miou": best_miou
        }, weights_dir / "checkpoint.pth")

        (run_dir / "last_epoch.txt").write_text(str(epoch + 1))

    # Final save
    to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(to_save.state_dict(), weights_dir / "final_model.pth")
    print("Final model saved ->", weights_dir / "final_model.pth")
    logger.info("Final model saved.")

    # Archive script for reproducibility
    try:
        shutil.copy(Path(__file__).resolve(), run_dir / "training_script.py")
    except Exception:
        pass


if __name__ == "__main__":
    main()
