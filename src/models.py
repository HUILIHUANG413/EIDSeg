# src/models.py
from __future__ import annotations
from typing import Tuple, Optional
import sys
from pathlib import Path

import torch
from torch import nn

from transformers import (
    AutoImageProcessor, AutoModelForSemanticSegmentation,
    SegformerImageProcessor, SegformerForSemanticSegmentation,
    Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation,
    OneFormerProcessor, OneFormerForUniversalSegmentation,
    #EomtForUniversalSegmentation
)

# ----------------------------- DeepLabV3+ support -----------------------------
def _maybe_add_deeplab_repo(repo_root: Optional[str] = None):
    """
    Ensure we can import `network.modeling` from the external repo.
    By default, assumes the repo folder is at project root: ../DeepLabV3Plus-Pytorch
    relative to this file (src/).
    """
    if repo_root is None:
        repo_root = str(Path(__file__).resolve().parents[1] / "DeepLabV3Plus-Pytorch")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

def load_deeplabv3plus(num_classes: int,
                       backbone: str = "resnet101",
                       output_stride: int = 16,
                       pretrained_backbone: bool = True):
    """
    Create a DeepLabV3+ model from the external repo.
    """
    _maybe_add_deeplab_repo()
    from network import modeling  # type: ignore
    # Available constructors in that repo:
    # - deeplabv3plus_resnet50 / _resnet101 / _xception / _mobilenet
    name = backbone.lower()
    if name in ("resnet50", "r50"):
        ctor = modeling.deeplabv3plus_resnet50
    elif name in ("resnet101", "r101"):
        ctor = modeling.deeplabv3plus_resnet101
    elif name in ("xception", "xcep"):
        ctor = modeling.deeplabv3plus_xception
    elif name in ("mobilenet", "mbv2", "mobilenetv2"):
        ctor = modeling.deeplabv3plus_mobilenet
    else:
        raise ValueError(f"Unsupported DeepLab backbone: {backbone}")

    model = ctor(num_classes=num_classes,
                 output_stride=output_stride,
                 pretrained_backbone=pretrained_backbone)
    return model
# src/models.py
def load_cityscapes_ckpt_if_given(model: nn.Module,
                                  ckpt_path: Optional[str],
                                  classifier_key_prefix: str = "classifier.classifier.3"):
    """
    Load Cityscapes pretrain from VainF DLv3+ repo.

    PyTorch 2.6 default is weights_only=True, which breaks older pickled checkpoints.
    We explicitly set weights_only=False (ONLY do this for trusted checkpoints).
    """
    if not ckpt_path:
        return

    # IMPORTANT: this allows legacy pickled objs (trusted file only)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Common key names in VainF releases
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # already a raw state_dict
        state_dict = ckpt
    else:
        raise RuntimeError(f"Unexpected checkpoint format: {type(ckpt)}")

    # Drop classifier head if num_classes mismatch
    try:
        out_ch = model.classifier.classifier[-1].out_channels
    except Exception:
        out_ch = None

    w_key = f"{classifier_key_prefix}.weight"
    b_key = f"{classifier_key_prefix}.bias"
    if out_ch is not None and w_key in state_dict:
        if state_dict[w_key].shape[0] != out_ch:
            state_dict.pop(w_key, None)
            state_dict.pop(b_key, None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[DeepLab] Loaded ckpt from {ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")




# ------------------------------- HF family loaders ----------------------------
def load_model_and_processor(model_name: str, num_classes: int, image_size: Tuple[int, int],
                             deeplab_backbone: Optional[str] = None,
                             deeplab_os: int = 16,
                             deeplab_cityscapes_ckpt: Optional[str] = None):
    """
    General factory. If model_name starts with 'deeplabv3plus', we build the DLv3+ path;
    otherwise we use the HF branches.
    Returns: (model, processor_or_None, model_type)
    model_type in {"oneformer","mask2former","segformer","beit","eomt","deeplabv3plus"}
    """
    lname = model_name.lower()

    # ----- DeepLabV3+ path -----
    if lname.startswith("deeplabv3plus"):
        model_type = "deeplabv3plus"
        backbone = deeplab_backbone or "resnet101"
        model = load_deeplabv3plus(num_classes=num_classes,
                                   backbone=backbone,
                                   output_stride=deeplab_os,
                                   pretrained_backbone=True)
        # optional Cityscapes pretrain
        if deeplab_cityscapes_ckpt:
            load_cityscapes_ckpt_if_given(model, deeplab_cityscapes_ckpt)

        processor = None  # handled in dataset via torchvision/albumentations
        return model, processor, model_type

    # ----- HF branches -----
    size_config = {"height": image_size[0], "width": image_size[1]}

    if "oneformer" in lname:
        model_type = "oneformer"
        processor = OneFormerProcessor.from_pretrained(model_name, size=size_config)
        model = OneFormerForUniversalSegmentation.from_pretrained(model_name)
        if hasattr(model.model.transformer_module.decoder, "class_embed"):
            in_feats = model.model.transformer_module.decoder.class_embed.in_features
            model.model.transformer_module.decoder.class_embed = nn.Linear(in_feats, num_classes)
        model.config.num_labels = num_classes

    elif "segformer" in lname:
        model_type = "segformer"
        processor = SegformerImageProcessor.from_pretrained(model_name, size=size_config)
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        model.config.num_labels = num_classes

    elif "beit" in lname:
        model_type = "beit"
        processor = AutoImageProcessor.from_pretrained(model_name, size=size_config)
        model = AutoModelForSemanticSegmentation.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        model.config.num_labels = num_classes

    elif "eomt" in lname:
        model_type = "eomt"
        processor = AutoImageProcessor.from_pretrained(
            model_name,
            crop_size=size_config,
            size={"shortest_edge": image_size[0], "longest_edge": image_size[1]}
        )
        model = EomtForUniversalSegmentation.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
        model.config.num_labels = num_classes

    else:
        model_type = "mask2former"
        processor = Mask2FormerImageProcessor.from_pretrained(model_name, size=size_config)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
        model.config.num_labels = num_classes
        if hasattr(model, "class_predictor") and isinstance(model.class_predictor, nn.Linear):
            model.class_predictor = nn.Linear(model.class_predictor.in_features, num_classes)

    return model, processor, model_type


def process_outputs_for_semantic(outputs, target_size: Tuple[int, int], model_type: str):
    """
    Unify logits → semantic predictions across architectures.
    Returns (semantic_logits[B,C,H,W], preds[B,H,W]).
    """
    if model_type in ("oneformer", "mask2former", "eomt"):
        class_logits = outputs.class_queries_logits   # (B, Q, C+1)
        mask_logits  = outputs.masks_queries_logits   # (B, Q, H, W)

        class_probs = torch.softmax(class_logits, dim=-1)
        mask_logits = torch.nn.functional.interpolate(
            mask_logits, size=target_size, mode="bilinear", align_corners=False
        )
        mask_probs = torch.sigmoid(mask_logits)
        semantic_logits = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
        preds = torch.argmax(semantic_logits, dim=1)

    elif model_type in ("segformer", "beit", "deeplabv3plus"):
        # All of these expose per-pixel logits directly: (B, C, H', W')
        semantic_logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        semantic_logits = torch.nn.functional.interpolate(
            semantic_logits, size=target_size, mode="bilinear", align_corners=False
        )
        preds = torch.argmax(semantic_logits, dim=1)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return semantic_logits, preds
