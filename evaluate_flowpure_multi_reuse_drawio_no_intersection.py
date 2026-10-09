#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-experiment standardized evaluation for FlowPure.

Features
- Builds or reuses standardized datasets for each experiment variant:
    <output>/standardized_eval/<experiment>__<variant>/
      clean/images, clean/labels
      adv/images, adv/labels
      denoised/images, denoised/labels
      _meta.json
- Evaluates with one or more detectors per experiment.
- Uses Ultralytics official model.val() for YOLOv8/YOLO26/RT-DETR.
- Supports Faster R-CNN via torchvision + pycocotools fallback.
- Computes lightweight metrics only:
    mAP50, mAP50_95, Precision, Recall, FPS
    MSE, MAE, PSNR
    optional SSIM
- Exports CSV / JSON / draw.io XML.

Notes
- DINOv2 weight path is kept in config for future extension, but not used by default
  in this lightweight script.
- If no Faster R-CNN checkpoint path is provided, the script falls back to
  torchvision COCO-pretrained Faster R-CNN.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import yaml
    YAML_OK = True
except Exception:
    yaml = None
    YAML_OK = False

try:
    import torch
    TORCH_OK = True
except Exception:
    torch = None
    TORCH_OK = False

try:
    from ultralytics import YOLO, RTDETR
    ULTRA_OK = True
except Exception:
    YOLO = None
    RTDETR = None
    ULTRA_OK = False

try:
    import torchvision
    from torchvision.transforms import functional as TVF
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
    TORCHVISION_OK = True
except Exception:
    torchvision = None
    TVF = None
    fasterrcnn_resnet50_fpn_v2 = None
    FasterRCNN_ResNet50_FPN_V2_Weights = None
    TORCHVISION_OK = False

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCO_OK = True
except Exception:
    COCO = None
    COCOeval = None
    PYCOCO_OK = False

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    SKIMAGE_OK = True
except Exception:
    skimage_ssim = None
    SKIMAGE_OK = False

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO80_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
    'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

DEFAULTS = {
    "clean_root": "/root/wangchao/FlowPure/datasets/coco",
    "label_root": "/root/wangchao/FlowPure/datasets/coco_labels/val2017",
    "weights": {
        "yolov8s": "/root/wangchao/FlowPure/yolov8s.pt",
        "yolov26s": "/root/wangchao/FlowPure/yolo26s.pt",
        "rtdetr": "/root/wangchao/FlowPure/rtdetr-l.pt",
        "fasterrcnn": None,
        "dinov2": "/root/wangchao/FlowPure/dinov2-vits14_model/model.safetensors",
    },
    "experiments": [
        {
            "experiment_group": "APGD_DINOv2",
            "attack": "APGD",
            "adv_root": "/root/wangchao/FlowPure/datasets/coco_dinov2_output/adv_images",
            "eval_detectors": ["yolov8s"],
            "variants": [
                {"image_set": "Perceptual", "role": "purified", "path": "/root/wangchao/FlowPure/fm_dinov2_finetune/purified_adv"},
            ],
        },
        {
            "experiment_group": "APGD_FasterRCNN",
            "attack": "APGD",
            "adv_root": "/root/wangchao/FlowPure/datasets/coco_fasterrcnn_output/adv_images",
            "eval_detectors": ["fasterrcnn"],
            "variants": [
                {"image_set": "Perceptual", "role": "purified", "path": "/root/wangchao/FlowPure/fm_fasterrcnn_finetune/purified_adv"},
            ],
        },
        {
            "experiment_group": "APGD_RTDETR",
            "attack": "APGD",
            "adv_root": "/root/wangchao/FlowPure/datasets/coco_rtdetr_output/adv_images",
            "eval_detectors": ["rtdetr"],
            "variants": [
                {"image_set": "Perceptual", "role": "purified", "path": "/root/wangchao/FlowPure/fm_rtdetr_finetune/purified_adv"},
            ],
        },
        {
            "experiment_group": "APGD_YOLOv26",
            "attack": "APGD",
            "adv_root": "/root/wangchao/FlowPure/datasets/coco_yolov26_output/adv_images",
            "eval_detectors": ["yolov26s"],
            "variants": [
                {"image_set": "Perceptual", "role": "purified", "path": "/root/wangchao/FlowPure/fm_yolov26_finetune/purified_adv"},
            ],
        },
        {
            "experiment_group": "Compare_YOLOv8s_APGD",
            "attack": "APGD",
            "adv_root": "/root/wangchao/FlowPure/datasets/coco_aa_output/adv_images",
            "eval_detectors": ["yolov8s"],
            "variants": [
                {"image_set": "GaussianBlur", "role": "baseline", "path": "/root/wangchao/FlowPure/eval_all/generated_baselines/baselines_yolov8s_apgd/GaussianBlur"},
                {"image_set": "MedianFilter", "role": "baseline", "path": "/root/wangchao/FlowPure/eval_all/generated_baselines/baselines_yolov8s_apgd/MedianFilter"},
                {"image_set": "JPEG75", "role": "baseline", "path": "/root/wangchao/FlowPure/eval_all/generated_baselines/baselines_yolov8s_apgd/JPEG75"},
                {"image_set": "VanillaFM", "role": "baseline", "path": "/root/wangchao/FlowPure/baseline_vanilla_fm/purified_vanilla_fm"},
                {"image_set": "CondDiff", "role": "baseline", "path": "/root/wangchao/FlowPure/baseline_cond_diff_v2/purified_cond_diffusion"},
                {"image_set": "Ours_Perceptual", "role": "purified", "path": "/root/wangchao/FlowPure/fm_vgg_finetune/purified_adv"},
            ],
        },
    ],
}


# -------------------------
# Utilities
# -------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_remove(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except FileNotFoundError:
        pass


def list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]
    files.sort()
    return files


def stem_map(root: Path) -> Dict[str, Path]:
    return {p.stem: p for p in list_images(root)}


def label_map(root: Path) -> Dict[str, Path]:
    if not root.exists():
        return {}
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
    files.sort()
    return {p.stem: p for p in files}


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8, copy=True)


def resize_to_match(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape[:2] == b.shape[:2]:
        return a, b
    b2 = np.array(Image.fromarray(b).resize((a.shape[1], a.shape[0]), Image.BILINEAR), dtype=np.uint8, copy=True)
    return a, b2


def image_to_float01(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 255.0


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = resize_to_match(a, b)
    return float(np.mean((image_to_float01(a) - image_to_float01(b)) ** 2))


def compute_mae(a: np.ndarray, b: np.ndarray) -> float:
    a, b = resize_to_match(a, b)
    return float(np.mean(np.abs(image_to_float01(a) - image_to_float01(b))))


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = compute_mse(a, b)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    if not SKIMAGE_OK:
        return float("nan")
    a, b = resize_to_match(a, b)
    return float(skimage_ssim(a, b, channel_axis=2, data_range=255))


def finite_or_nan(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def box_xywh_norm_to_xyxy_abs(xc, yc, bw, bh, W, H):
    x1 = max(0.0, (xc - bw / 2.0) * W)
    y1 = max(0.0, (yc - bh / 2.0) * H)
    x2 = min(float(W), (xc + bw / 2.0) * W)
    y2 = min(float(H), (yc + bh / 2.0) * H)
    return [x1, y1, x2, y2]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def link_or_copy(src: Path, dst: Path, prefer_symlink: bool = True) -> None:
    safe_remove(dst)
    if prefer_symlink:
        try:
            os.symlink(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


# -------------------------
# Standardized dataset reuse/build
# -------------------------

def build_standardized_mode(src_image_root: Path, label_root: Path, out_mode_root: Path, allowed_stems: Sequence[str], prefer_symlink: bool = True) -> int:
    img_dir = ensure_dir(out_mode_root / "images")
    lab_dir = ensure_dir(out_mode_root / "labels")
    src_imgs = stem_map(src_image_root)
    src_labs = label_map(label_root)
    n = 0
    for stem in allowed_stems:
        img_path = src_imgs.get(stem)
        lab_path = src_labs.get(stem)
        if img_path is None or lab_path is None:
            continue
        link_or_copy(img_path, img_dir / img_path.name, prefer_symlink=prefer_symlink)
        link_or_copy(lab_path, lab_dir / lab_path.name, prefer_symlink=prefer_symlink)
        n += 1
    return n


def common_stems_for_triplet(clean_root: Path, adv_root: Path, den_root: Path, label_root: Path) -> List[str]:
    clean = set(stem_map(clean_root).keys())
    adv = set(stem_map(adv_root).keys())
    den = set(stem_map(den_root).keys())
    labs = set(label_map(label_root).keys())
    return sorted(clean & adv & den & labs)

def build_standardized_mode_all(src_image_root: Path, label_root: Path, out_mode_root: Path, prefer_symlink: bool = True, sample_limit: Optional[int] = None, create_empty_label: bool = True) -> int:
    img_dir = ensure_dir(out_mode_root / "images")
    lab_dir = ensure_dir(out_mode_root / "labels")
    src_imgs = list_images(src_image_root)
    if sample_limit is not None:
        src_imgs = src_imgs[:sample_limit]
    src_labs = label_map(label_root)
    n = 0
    for img_path in src_imgs:
        link_or_copy(img_path, img_dir / img_path.name, prefer_symlink=prefer_symlink)
        dst_lab = lab_dir / f"{img_path.stem}.txt"
        lab_path = src_labs.get(img_path.stem)
        if lab_path is not None:
            link_or_copy(lab_path, dst_lab, prefer_symlink=prefer_symlink)
        elif create_empty_label:
            ensure_dir(dst_lab.parent)
            dst_lab.write_text("", encoding="utf-8")
        n += 1
    return n


def standardized_dir_counts(work_root: Path) -> Dict[str, int]:
    return {
        "clean_images": len(list_images(work_root / "clean" / "images")),
        "adv_images": len(list_images(work_root / "adv" / "images")),
        "denoised_images": len(list_images(work_root / "denoised" / "images")),
        "clean_labels": len(label_map(work_root / "clean" / "labels")),
        "adv_labels": len(label_map(work_root / "adv" / "labels")),
        "denoised_labels": len(label_map(work_root / "denoised" / "labels")),
    }


def standardized_triplet_exists(work_root: Path) -> Tuple[bool, int]:
    counts = standardized_dir_counts(work_root)
    needed = [
        counts["clean_images"], counts["adv_images"], counts["denoised_images"],
        counts["clean_labels"], counts["adv_labels"], counts["denoised_labels"],
    ]
    ok = all(x > 0 for x in needed)
    n = min(needed) if ok else 0
    return ok, n


def build_meta_dict(clean_root: Path, adv_root: Path, den_root: Path, label_root: Path, sample_limit: Optional[int], prefer_symlink: bool) -> Dict:
    return {
        "clean_root": str(clean_root),
        "adv_root": str(adv_root),
        "den_root": str(den_root),
        "label_root": str(label_root),
        "sample_limit": sample_limit,
        "prefer_symlink": bool(prefer_symlink),
    }


def meta_matches(work_root: Path, expected_meta: Dict) -> bool:
    meta_path = work_root / "_meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
    except Exception:
        return False
    for k, v in expected_meta.items():
        if meta.get(k) != v:
            return False
    return True


def save_meta(work_root: Path, meta: Dict, common_count: int) -> None:
    meta_path = work_root / "_meta.json"
    to_save = dict(meta)
    to_save["common_count"] = int(common_count)
    to_save["counts"] = standardized_dir_counts(work_root)
    json.dump(to_save, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def prepare_standardized_triplet_dataset(
    work_root: Path,
    clean_root: Path,
    adv_root: Path,
    den_root: Path,
    label_root: Path,
    sample_limit: Optional[int] = None,
    prefer_symlink: bool = True,
    force_rebuild: bool = False,
) -> Tuple[Path, int, bool]:
    expected_meta = build_meta_dict(clean_root, adv_root, den_root, label_root, sample_limit, prefer_symlink)
    expected_meta["build_policy"] = "link_all_images_no_intersection"
    if not force_rebuild:
        ok, n_existing = standardized_triplet_exists(work_root)
        if ok and meta_matches(work_root, expected_meta):
            return work_root, n_existing, True
        if ok and not (work_root / "_meta.json").exists():
            return work_root, n_existing, True

    safe_remove(work_root)
    ensure_dir(work_root)

    # Do not take clean/adv/denoised intersection here.
    # Link every image from each provided folder directly.
    n_clean = build_standardized_mode_all(clean_root, label_root, work_root / "clean", prefer_symlink=prefer_symlink, sample_limit=sample_limit)
    n_adv = build_standardized_mode_all(adv_root, label_root, work_root / "adv", prefer_symlink=prefer_symlink, sample_limit=sample_limit)
    n_den = build_standardized_mode_all(den_root, label_root, work_root / "denoised", prefer_symlink=prefer_symlink, sample_limit=sample_limit)

    n = min(n_clean, n_adv, n_den)
    save_meta(work_root, expected_meta, n)
    return work_root, n, False


# -------------------------
# Metrics containers
# -------------------------
@dataclass
class EvalMetrics:
    mAP50: float = float("nan")
    mAP50_95: float = float("nan")
    Precision: float = float("nan")
    Recall: float = float("nan")
    FPS: float = float("nan")
    num_images_eval: int = 0


@dataclass
class ImageMetrics:
    num_common_pairs: int = 0
    MSE: float = float("nan")
    MAE: float = float("nan")
    PSNR: float = float("nan")
    SSIM: float = float("nan")


# -------------------------
# Detector evaluators
# -------------------------
class BaseEvaluator:
    def evaluate_mode(self, dataset_mode_root: Path, device_spec: str, imgsz: int, conf: float, iou: float, verbose: bool = False) -> EvalMetrics:
        raise NotImplementedError


class UltralyticsValEvaluator(BaseEvaluator):
    def __init__(self, model_kind: str, weight_path: str):
        if not ULTRA_OK:
            raise RuntimeError("ultralytics is not installed")
        self.model_kind = model_kind
        if model_kind == "rtdetr":
            self.model = RTDETR(weight_path)
        else:
            self.model = YOLO(weight_path)
        self.names_obj = getattr(self.model, "names", None)

    def make_yaml_text(self, dataset_root: Path) -> str:
        names_obj = self.names_obj
        names_list = None
        if isinstance(names_obj, dict):
            try:
                names_list = [names_obj[k] for k in sorted(names_obj.keys())]
            except Exception:
                names_list = list(names_obj.values())
        elif isinstance(names_obj, (list, tuple)):
            names_list = list(names_obj)

        if YAML_OK:
            data = {
                "path": str(dataset_root),
                "train": "images",
                "val": "images",
                "names": names_list if names_list is not None else names_obj,
            }
            return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

        lines = [f"path: {dataset_root}", "train: images", "val: images", "names:"]
        if names_list is not None:
            for i, name in enumerate(names_list):
                lines.append(f"  {i}: {name}")
        elif isinstance(names_obj, dict):
            for k, v in names_obj.items():
                lines.append(f"  {k}: {v}")
        else:
            for i, name in enumerate(COCO80_NAMES):
                lines.append(f"  {i}: {name}")
        return "\n".join(lines) + "\n"

    def evaluate_mode(self, dataset_mode_root: Path, device_spec: str, imgsz: int, conf: float, iou: float, verbose: bool = False) -> EvalMetrics:
        img_dir = dataset_mode_root / "images"
        image_files = list_images(img_dir)
        n = len(image_files)
        if n == 0:
            return EvalMetrics(num_images_eval=0)

        yaml_path = dataset_mode_root / "temp_eval.yaml"
        yaml_path.write_text(self.make_yaml_text(dataset_mode_root), encoding="utf-8")

        if TORCH_OK and isinstance(device_spec, str) and device_spec.startswith("cuda") and torch.cuda.is_available():
            try:
                dev_index = 0
                if ":" in device_spec:
                    dev_index = int(device_spec.split(":")[1])
                with torch.cuda.device(dev_index):
                    torch.cuda.synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    results = self.model.val(
                        data=str(yaml_path), device=device_spec, imgsz=imgsz, conf=conf, iou=iou,
                        save_json=False, plots=False, verbose=verbose,
                    )
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed_s = start_event.elapsed_time(end_event) / 1000.0
            except Exception:
                t0 = time.perf_counter()
                results = self.model.val(
                    data=str(yaml_path), device=device_spec, imgsz=imgsz, conf=conf, iou=iou,
                    save_json=False, plots=False, verbose=verbose,
                )
                elapsed_s = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            results = self.model.val(
                data=str(yaml_path), device=device_spec, imgsz=imgsz, conf=conf, iou=iou,
                save_json=False, plots=False, verbose=verbose,
            )
            elapsed_s = time.perf_counter() - t0

        md = getattr(results, "results_dict", {}) or {}
        metrics = EvalMetrics(
            mAP50=finite_or_nan(md.get("metrics/mAP50(B)", md.get("metrics/mAP50", float("nan")))),
            mAP50_95=finite_or_nan(md.get("metrics/mAP50-95(B)", md.get("metrics/mAP50-95", float("nan")))),
            Precision=finite_or_nan(md.get("metrics/precision(B)", md.get("metrics/precision", float("nan")))),
            Recall=finite_or_nan(md.get("metrics/recall(B)", md.get("metrics/recall", float("nan")))),
            FPS=float(n / elapsed_s) if elapsed_s > 1e-9 else float("nan"),
            num_images_eval=n,
        )

        try:
            yaml_path.unlink()
        except Exception:
            pass
        if Path("runs").exists():
            shutil.rmtree("runs", ignore_errors=True)
        return metrics


class FasterRCNNEvaluator(BaseEvaluator):
    def __init__(self, ckpt_path: Optional[str] = None):
        if not TORCHVISION_OK:
            raise RuntimeError("torchvision detection is not installed")
        self.ckpt_path = ckpt_path
        if ckpt_path and Path(ckpt_path).exists():
            self.model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=91)
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)
        else:
            self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        self.model.eval()

    def _build_gt(self, mode_root: Path) -> Tuple[Path, Dict, Dict[str, int]]:
        gt_json = mode_root / "_gt_coco.json"
        images = list_images(mode_root / "images")
        categories = [{"id": i + 1, "name": name} for i, name in enumerate(COCO80_NAMES)]
        img_entries, ann_entries = [], []
        img_id_map = {}
        ann_id = 1
        for img_id, img_path in enumerate(images, start=1):
            arr = load_rgb(img_path)
            H, W = arr.shape[:2]
            img_entries.append({"id": img_id, "file_name": img_path.name, "width": W, "height": H})
            img_id_map[img_path.stem] = img_id
            lab = mode_root / "labels" / f"{img_path.stem}.txt"
            if lab.exists():
                for line in open(lab, "r", encoding="utf-8"):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                    box = box_xywh_norm_to_xyxy_abs(xc, yc, bw, bh, W, H)
                    xywh = xyxy_to_xywh(box)
                    ann_entries.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls + 1,
                        "bbox": xywh,
                        "area": max(0.0, xywh[2]) * max(0.0, xywh[3]),
                        "iscrowd": 0,
                    })
                    ann_id += 1
        json.dump({"images": img_entries, "annotations": ann_entries, "categories": categories}, open(gt_json, "w", encoding="utf-8"))
        gt_data = {"annotations": ann_entries}
        return gt_json, gt_data, img_id_map

    def _predict(self, mode_root: Path, device_spec: str) -> Tuple[List[Path], List[Dict], float]:
        image_paths = list_images(mode_root / "images")
        n = len(image_paths)
        if n == 0:
            return [], [], float("nan")
        device = torch.device(device_spec if TORCH_OK else "cpu")
        self.model.to(device)
        preds = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for p in image_paths:
                img = Image.open(p).convert("RGB")
                tensor = TVF.to_tensor(img).to(device)
                out = self.model([tensor])[0]
                boxes = out["boxes"].detach().cpu().numpy().astype(np.float32)
                scores = out["scores"].detach().cpu().numpy().astype(np.float32)
                labels = out["labels"].detach().cpu().numpy().astype(np.int64) - 1
                valid = (labels >= 0) & (labels < 80)
                preds.append({"boxes": boxes[valid], "scores": scores[valid], "labels": labels[valid]})
        elapsed = time.perf_counter() - t0
        fps = float(n / elapsed) if elapsed > 1e-9 else float("nan")
        return image_paths, preds, fps

    def _save_pred_json(self, mode_root: Path, image_paths: List[Path], preds: List[Dict], image_id_map: Dict[str, int]) -> Path:
        pred_json = mode_root / "_pred_fasterrcnn.json"
        dets = []
        for p, pred in zip(image_paths, preds):
            image_id = image_id_map[p.stem]
            for box, score, label in zip(pred["boxes"], pred["scores"], pred["labels"]):
                x1, y1, x2, y2 = map(float, box.tolist())
                dets.append({
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })
        json.dump(dets, open(pred_json, "w", encoding="utf-8"))
        return pred_json

    def _run_coco_eval(self, gt_json: Path, pred_json: Path) -> Tuple[float, float]:
        if not PYCOCO_OK:
            print("[WARN] pycocotools not available, Faster R-CNN mAP50/mAP50_95 will be NaN while custom Precision/Recall can still be computed.")
            return float("nan"), float("nan")
        try:
            coco_gt = COCO(str(gt_json))
            coco_dt = coco_gt.loadRes(str(pred_json)) if pred_json.exists() else coco_gt.loadRes([])
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
            ap5095 = float(coco_eval.stats[0])
            ap50 = float(coco_eval.stats[1])
            return ap50, ap5095
        except Exception as e:
            print(f"[WARN] Faster R-CNN COCO evaluation failed: {e}")
            return float("nan"), float("nan")

    @staticmethod
    def _iou_xyxy(a, b) -> float:
        xA = max(a[0], b[0]); yA = max(a[1], b[1]); xB = min(a[2], b[2]); yB = min(a[3], b[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        areaB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

    def _compute_pr_iou50(self, mode_root: Path, image_paths: List[Path], preds: List[Dict]) -> Tuple[float, float]:
        # simple global precision / recall at IoU 0.5
        gt_by_stem = {}
        for p in image_paths:
            arr = load_rgb(p); H, W = arr.shape[:2]
            labs = []
            lab_path = mode_root / "labels" / f"{p.stem}.txt"
            if lab_path.exists():
                for line in open(lab_path, "r", encoding="utf-8"):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                    labs.append((cls, box_xywh_norm_to_xyxy_abs(xc, yc, bw, bh, W, H)))
            gt_by_stem[p.stem] = labs

        TP = FP = FN = 0
        for p, pred in zip(image_paths, preds):
            gt_list = gt_by_stem[p.stem]
            used = [False] * len(gt_list)
            order = np.argsort(-pred["scores"]) if len(pred["scores"]) else np.array([], dtype=int)
            for idx in order:
                box = pred["boxes"][idx]
                cls = int(pred["labels"][idx])
                best_iou, best_j = 0.0, -1
                for j, (gt_cls, gt_box) in enumerate(gt_list):
                    if used[j] or gt_cls != cls:
                        continue
                    iou = self._iou_xyxy(box, gt_box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= 0.5 and best_j >= 0:
                    used[best_j] = True
                    TP += 1
                else:
                    FP += 1
            FN += sum(1 for u in used if not u)
        precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
        recall = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
        return float(precision), float(recall)

    def evaluate_mode(self, dataset_mode_root: Path, device_spec: str, imgsz: int, conf: float, iou: float, verbose: bool = False) -> EvalMetrics:
        image_paths = list_images(dataset_mode_root / "images")
        if not image_paths:
            return EvalMetrics(num_images_eval=0)
        gt_json, gt_data, image_id_map = self._build_gt(dataset_mode_root)
        image_paths, preds, fps = self._predict(dataset_mode_root, device_spec)
        pred_json = self._save_pred_json(dataset_mode_root, image_paths, preds, image_id_map)
        ap50, ap5095 = self._run_coco_eval(gt_json, pred_json)
        precision, recall = self._compute_pr_iou50(dataset_mode_root, image_paths, preds)
        return EvalMetrics(mAP50=ap50, mAP50_95=ap5095, Precision=precision, Recall=recall, FPS=fps, num_images_eval=len(image_paths))


def build_evaluators(weights_cfg: Dict, needed_detectors: Sequence[str]) -> Dict[str, BaseEvaluator]:
    evaluators = {}
    for det in needed_detectors:
        if det == "yolov8s":
            evaluators[det] = UltralyticsValEvaluator("yolo", weights_cfg[det])
        elif det == "yolov26s":
            evaluators[det] = UltralyticsValEvaluator("yolo", weights_cfg[det])
        elif det == "rtdetr":
            evaluators[det] = UltralyticsValEvaluator("rtdetr", weights_cfg[det])
        elif det == "fasterrcnn":
            evaluators[det] = FasterRCNNEvaluator(weights_cfg.get(det))
        else:
            raise ValueError(f"Unsupported detector: {det}")
    return evaluators


# -------------------------
# Lightweight image metrics / derived
# -------------------------

def compute_pair_metrics(clean_mode_root: Path, other_mode_root: Path, include_ssim: bool = False) -> ImageMetrics:
    clean_map_ = stem_map(clean_mode_root / "images")
    other_map_ = stem_map(other_mode_root / "images")
    common = sorted(set(clean_map_.keys()) & set(other_map_.keys()))
    if not common:
        return ImageMetrics(num_common_pairs=0)

    mse_vals, mae_vals, psnr_vals, ssim_vals = [], [], [], []
    for stem in common:
        a = load_rgb(clean_map_[stem])
        b = load_rgb(other_map_[stem])
        mse_vals.append(compute_mse(a, b))
        mae_vals.append(compute_mae(a, b))
        psnr_vals.append(compute_psnr(a, b))
        if include_ssim:
            ssim_vals.append(compute_ssim(a, b))

    def avg(vals):
        return float(np.mean(vals)) if vals else float("nan")

    return ImageMetrics(
        num_common_pairs=len(common),
        MSE=avg(mse_vals),
        MAE=avg(mae_vals),
        PSNR=avg(psnr_vals),
        SSIM=avg(ssim_vals) if include_ssim else float("nan"),
    )


def derive_metrics(clean_map50: float, adv_map50: float, cur_map50: float) -> Dict[str, float]:
    out = {
        "mAP_Drop": float("nan"),
        "Relative_mAP_Drop": float("nan"),
        "Recovery_Rate": float("nan"),
        "Clean_Retention": float("nan"),
        "Gain_over_Adv": float("nan"),
        "Improvement_over_Adv_pct": float("nan"),
    }
    if np.isfinite(clean_map50) and np.isfinite(cur_map50):
        out["mAP_Drop"] = float(clean_map50 - cur_map50)
        out["Clean_Retention"] = float(cur_map50 / clean_map50) if abs(clean_map50) > 1e-12 else float("nan")
        out["Relative_mAP_Drop"] = float((clean_map50 - cur_map50) / clean_map50) if abs(clean_map50) > 1e-12 else float("nan")
    if np.isfinite(adv_map50) and np.isfinite(cur_map50):
        out["Gain_over_Adv"] = float(cur_map50 - adv_map50)
        out["Improvement_over_Adv_pct"] = float((cur_map50 - adv_map50) / adv_map50) if abs(adv_map50) > 1e-12 else float("nan")
    if np.isfinite(clean_map50) and np.isfinite(adv_map50) and np.isfinite(cur_map50):
        denom = clean_map50 - adv_map50
        out["Recovery_Rate"] = float((cur_map50 - adv_map50) / denom) if abs(denom) > 1e-12 else float("nan")
    return out


# -------------------------
# draw.io XML
# -------------------------

def xml_escape(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")


class DrawioDoc:
    def __init__(self):
        self.pages = []

    def add_page(self, name: str, cells: List[str]):
        inner = "".join(cells)
        graph = (
            '<mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" '
            'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1920" pageHeight="1080" '
            'math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
            f'{inner}</root></mxGraphModel>'
        )
        self.pages.append(f'<diagram name="{xml_escape(name)}">{graph}</diagram>')

    def save(self, path: Path):
        txt = '<?xml version="1.0" encoding="UTF-8"?><mxfile host="app.diagrams.net" version="24.7.17">' + "".join(self.pages) + '</mxfile>'
        path.write_text(txt, encoding="utf-8")


def cell_rect(cell_id: int, x: float, y: float, w: float, h: float, value: str, style: str) -> str:
    return (
        f'<mxCell id="{cell_id}" value="{xml_escape(value)}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
    )


def build_table_page(title: str, headers: List[str], rows: List[List[str]]) -> List[str]:
    cells = []
    cid = 2
    cells.append(cell_rect(cid, 20, 20, 1600, 40, title, 'rounded=1;whiteSpace=wrap;html=1;fontStyle=1;fontSize=20;fillColor=#dae8fc;strokeColor=#6c8ebf;'))
    cid += 1
    x0, y0 = 20, 80
    colw = [120] * len(headers)
    colw[0] = 160
    colw[1] = 120
    colw[2] = 150
    rowh = 28

    x = x0
    for h, w in zip(headers, colw):
        cells.append(cell_rect(cid, x, y0, w, rowh, h, 'rounded=0;whiteSpace=wrap;html=1;fontStyle=1;fontSize=12;fillColor=#f5f5f5;strokeColor=#666666;'))
        cid += 1
        x += w

    for i, row in enumerate(rows):
        x = x0
        y = y0 + rowh * (i + 1)
        for j, (v, w) in enumerate(zip(row, colw)):
            fill = '#ffffff' if i % 2 == 0 else '#fafafa'
            if j <= 2:
                fill = '#fff2cc'
            cells.append(cell_rect(cid, x, y, w, rowh, v, f'rounded=0;whiteSpace=wrap;html=1;fontSize=11;fillColor={fill};strokeColor=#c0c0c0;'))
            cid += 1
            x += w
    return cells


def build_bar_page(title: str, labels: List[str], values: List[float], y_label: str) -> List[str]:
    cells = []
    cid = 2
    cells.append(cell_rect(cid, 20, 20, 1200, 40, f"{title} ({y_label})", 'rounded=1;whiteSpace=wrap;html=1;fontStyle=1;fontSize=20;fillColor=#dae8fc;strokeColor=#6c8ebf;'))
    cid += 1
    chart_x, chart_y, chart_w, chart_h = 80, 120, 1000, 500
    cells.append(cell_rect(cid, chart_x, chart_y, chart_w, chart_h, "", 'rounded=0;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#666666;'))
    cid += 1
    finite_vals = [v for v in values if np.isfinite(v)]
    max_val = max(finite_vals + [1e-6])
    n = len(labels)
    bar_gap = 20
    usable_w = chart_w - 60
    bar_w = max(20, (usable_w - bar_gap * max(n - 1, 0)) / max(n, 1))
    x = chart_x + 40
    base_y = chart_y + chart_h - 40
    for lab, val in zip(labels, values):
        v = 0.0 if not np.isfinite(val) else float(val)
        bh = 0.0 if max_val <= 0 else (v / max_val) * (chart_h - 90)
        cells.append(cell_rect(cid, x, base_y - bh, bar_w, bh, "", 'rounded=0;whiteSpace=wrap;html=1;fillColor=#6fa8dc;strokeColor=#3d85c6;'))
        cid += 1
        cells.append(cell_rect(cid, x - 10, base_y + 8, bar_w + 20, 28, lab, 'rounded=0;whiteSpace=wrap;html=1;fontSize=10;align=center;verticalAlign=top;strokeColor=none;fillColor=none;'))
        cid += 1
        cells.append(cell_rect(cid, x - 10, base_y - bh - 24, bar_w + 20, 20, f"{v:.4f}" if np.isfinite(val) else "NaN", 'rounded=0;whiteSpace=wrap;html=1;fontSize=10;align=center;strokeColor=none;fillColor=none;'))
        cid += 1
        x += bar_w + bar_gap
    return cells


def fmt(x) -> str:
    if x is None:
        return ""
    try:
        xf = float(x)
        if math.isnan(xf):
            return "NaN"
        if math.isinf(xf):
            return "Inf"
        return f"{xf:.6f}"
    except Exception:
        return str(x)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default="./eval_flowpure_multi")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=8)  # reserved
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--sample-limit", type=int, default=None)
    ap.add_argument("--include-ssim", action="store_true")
    ap.add_argument("--skip-image-metrics", action="store_true")
    ap.add_argument("--prefer-copy", action="store_true")
    ap.add_argument("--config-json", type=str, default=None)
    ap.add_argument("--verbose-val", action="store_true")
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(json.dumps(DEFAULTS))
    if args.config_json:
        cfg = json.load(open(args.config_json, "r", encoding="utf-8"))

    out_dir = ensure_dir(Path(args.output_dir))
    work_dir = ensure_dir(out_dir / "standardized_eval")
    drawio_dir = ensure_dir(out_dir / "drawio_xml")

    # collect detectors actually needed
    needed_detectors = []
    for exp in cfg["experiments"]:
        for det in exp.get("eval_detectors", ["yolov8s"]):
            if det not in needed_detectors:
                needed_detectors.append(det)

    print(f"[INFO] Output: {out_dir}")
    print(f"[INFO] Building evaluators for: {needed_detectors}")
    evaluators = build_evaluators(cfg["weights"], needed_detectors)

    all_rows = []
    for exp in cfg["experiments"]:
        exp_group = exp["experiment_group"]
        attack = exp["attack"]
        clean_root = Path(cfg["clean_root"])
        adv_root = Path(exp["adv_root"])
        eval_detectors = exp.get("eval_detectors", ["yolov8s"])
        for variant in exp["variants"]:
            image_set = variant["image_set"]
            den_root = Path(variant["path"])
            role = variant.get("role", "purified")
            exp_key = f"{exp_group}__{image_set}"
            print(f"\n[INFO] Checking standardized dataset: {exp_key}")
            triplet_root, n_common, reused = prepare_standardized_triplet_dataset(
                work_dir / exp_key,
                clean_root=clean_root,
                adv_root=adv_root,
                den_root=den_root,
                label_root=Path(cfg["label_root"]),
                sample_limit=args.sample_limit,
                prefer_symlink=(not args.prefer_copy),
                force_rebuild=args.force_rebuild,
            )
            if reused:
                print(f"[INFO] Reusing existing standardized dataset: {triplet_root}")
            else:
                print(f"[INFO] Built standardized dataset: {triplet_root}")
            print(f"[INFO] Common paired samples: {n_common}")

            for det_name in eval_detectors:
                evaluator = evaluators[det_name]
                print(f"[INFO] Evaluating detectors for {exp_key}: {det_name}")
                mode_rows = {}
                for mode_name, role_name in [("clean", "clean"), ("adv", "adv"), ("denoised", role)]:
                    mode_root = triplet_root / mode_name
                    print(f"[INFO] Eval -> detector={det_name} mode={mode_name}")
                    eval_metrics = evaluator.evaluate_mode(
                        dataset_mode_root=mode_root,
                        device_spec=args.device,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        iou=args.iou,
                        verbose=args.verbose_val,
                    )
                    image_metrics = ImageMetrics()
                    if not args.skip_image_metrics and mode_name != "clean":
                        image_metrics = compute_pair_metrics(triplet_root / "clean", mode_root, include_ssim=args.include_ssim)
                    row = {
                        "experiment_group": exp_group,
                        "attack": attack,
                        "detector": det_name,
                        "image_set": "Clean" if mode_name == "clean" else "Adv" if mode_name == "adv" else image_set,
                        "role": role_name,
                        "path": str(clean_root if mode_name == "clean" else adv_root if mode_name == "adv" else den_root),
                        "num_images_eval": eval_metrics.num_images_eval,
                        "num_common_pairs": image_metrics.num_common_pairs if mode_name != "clean" else n_common,
                        "mAP50": eval_metrics.mAP50,
                        "mAP50_95": eval_metrics.mAP50_95,
                        "Precision": eval_metrics.Precision,
                        "Recall": eval_metrics.Recall,
                        "FPS": eval_metrics.FPS,
                        "MSE": image_metrics.MSE if mode_name != "clean" else float("nan"),
                        "MAE": image_metrics.MAE if mode_name != "clean" else float("nan"),
                        "PSNR": image_metrics.PSNR if mode_name != "clean" else float("nan"),
                        "SSIM": image_metrics.SSIM if (mode_name != "clean" and args.include_ssim) else float("nan"),
                    }
                    mode_rows[mode_name] = row
                clean_map50 = mode_rows["clean"]["mAP50"]
                adv_map50 = mode_rows["adv"]["mAP50"]
                for mode_name in ["clean", "adv", "denoised"]:
                    row = mode_rows[mode_name]
                    row.update(derive_metrics(clean_map50, adv_map50, row["mAP50"]))
                    all_rows.append(row)

    headers = [
        "experiment_group", "attack", "detector", "image_set", "role", "path",
        "num_images_eval", "num_common_pairs",
        "mAP50", "mAP50_95", "Precision", "Recall", "FPS",
        "MSE", "MAE", "PSNR", "SSIM",
        "mAP_Drop", "Relative_mAP_Drop", "Recovery_Rate",
        "Clean_Retention", "Gain_over_Adv", "Improvement_over_Adv_pct"
    ]
    csv_path = out_dir / "all_results.csv"
    json_path = out_dir / "all_results.json"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in all_rows:
            vals = []
            for h in headers:
                v = row.get(h, "")
                if isinstance(v, str):
                    vals.append('"' + v.replace('"', '""') + '"')
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")
    json.dump(all_rows, json_path.open("w", encoding="utf-8"), ensure_ascii=False, indent=2)

    doc = DrawioDoc()
    table_headers = ["experiment_group", "detector", "image_set", "mAP50", "mAP50_95", "Precision", "Recall", "FPS", "MSE", "MAE", "PSNR"]
    table_rows = []
    for r in all_rows:
        table_rows.append([str(r[h]) if h in {"experiment_group", "detector", "image_set"} else fmt(r[h]) for h in table_headers])
    doc.add_page("Summary_Table", build_table_page("FlowPure Multi-Experiment Evaluation", table_headers, table_rows))

    group_keys = sorted(set((r["experiment_group"], r["detector"]) for r in all_rows))
    for grp, det in group_keys:
        grp_rows = [r for r in all_rows if r["experiment_group"] == grp and r["detector"] == det and r["image_set"] != "Clean"]
        labels = [r["image_set"] for r in grp_rows]
        if labels:
            prefix = f"{grp}_{det}"
            doc.add_page(f"{prefix}_mAP50", build_bar_page(f"{grp} / {det} mAP50", labels, [r["mAP50"] for r in grp_rows], "mAP50"))
            doc.add_page(f"{prefix}_Precision", build_bar_page(f"{grp} / {det} Precision", labels, [r["Precision"] for r in grp_rows], "Precision"))
            doc.add_page(f"{prefix}_Recall", build_bar_page(f"{grp} / {det} Recall", labels, [r["Recall"] for r in grp_rows], "Recall"))
            doc.add_page(f"{prefix}_PSNR", build_bar_page(f"{grp} / {det} PSNR", labels, [r["PSNR"] for r in grp_rows], "PSNR"))

    drawio_path = drawio_dir / "flowpure_multi_eval.drawio.xml"
    doc.save(drawio_path)

    print(f"\n[INFO] Saved CSV: {csv_path}")
    print(f"[INFO] Saved JSON: {json_path}")
    print(f"[INFO] Saved draw.io XML: {drawio_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

# python evaluate_flowpure_multi_reuse_drawio_no_intersection.py \
#   --output-dir ./eval_flowpure_multi \
#   --device cuda:0 \
#   --imgsz 640 