# 最新版评估yolov8s的脚本，一次性评估全部实验
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight standardized evaluation for FlowPure experiments.

Updated behavior:
- Reuse existing standardized datasets by default.
- Only rebuild when:
  1) standardized dataset does not exist, or
  2) metadata/source mismatch is detected, or
  3) --force-rebuild is given.

Standardized dataset layout:
    <exp_root>/
      clean/images, clean/labels
      adv/images, adv/labels
      denoised/images, denoised/labels
      _meta.json
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
    from ultralytics import YOLO
    ULTRA_OK = True
except Exception:
    YOLO = None
    ULTRA_OK = False

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    SKIMAGE_OK = True
except Exception:
    skimage_ssim = None
    SKIMAGE_OK = False

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULTS = {
    "clean_root": "/root/wangchao/FlowPure/datasets/coco",
    "label_root": "/root/wangchao/FlowPure/datasets/coco_labels/val2017",
    "yolo_weight": "/root/wangchao/FlowPure/yolov8s.pt",
    "experiments": [
        {
            "experiment_group": "APGD_Ablation",
            "attack": "APGD",
            "adv_root": "/root/wangchao/FlowPure/datasets/coco_aa_output/adv_images",
            "variants": [
                {"image_set": "Base_NoPerceptual", "role": "purified", "path": "/root/wangchao/FlowPure/fm_aa_out/purified_results"},
                {"image_set": "BG_Protected", "role": "purified", "path": "/root/wangchao/FlowPure/fm_aa_out/purified_bg_protected"},
                {"image_set": "Full_Aligned", "role": "purified", "path": "/root/wangchao/FlowPure/fm_aa_out/purified_full_aligned"},
                {"image_set": "Final_Sprint", "role": "purified", "path": "/root/wangchao/FlowPure/fm_aa_out/purified_final_sprint"},
                {"image_set": "Heun", "role": "purified", "path": "/root/wangchao/FlowPure/fm_aa_out/purified_results_heun"},
                {"image_set": "Perceptual", "role": "purified", "path": "/root/wangchao/FlowPure/fm_vgg_finetune/purified_adv"},
                {"image_set": "StopTime_0.85", "role": "purified", "path": "/root/wangchao/FlowPure/fm_vgg_finetune/strong_purify_st0.85_step30"},
                {"image_set": "StopTime_0.95", "role": "purified", "path": "/root/wangchao/FlowPure/fm_vgg_finetune/strong_purify_st0.95_step50"},
            ],
        },
    ],
}

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
        x = float(x)
        return x
    except Exception:
        return float("nan")

def link_or_copy(src: Path, dst: Path, prefer_symlink: bool = True) -> None:
    safe_remove(dst)
    if prefer_symlink:
        try:
            os.symlink(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def make_yaml_text(dataset_root: Path, names_obj) -> str:
    names_list = None
    if isinstance(names_obj, dict):
        try:
            names_list = [names_obj[k] for k in sorted(names_obj.keys())]
        except Exception:
            names_list = list(names_obj.values())
    elif isinstance(names_obj, (list, tuple)):
        names_list = list(names_obj)
    else:
        names_list = None

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
        lines.append("  0: object")
    return "\n".join(lines) + "\n"

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
    if not force_rebuild:
        ok, n_existing = standardized_triplet_exists(work_root)
        if ok and meta_matches(work_root, expected_meta):
            return work_root, n_existing, True
        if ok and not (work_root / "_meta.json").exists():
            return work_root, n_existing, True

    safe_remove(work_root)
    ensure_dir(work_root)
    common = common_stems_for_triplet(clean_root, adv_root, den_root, label_root)
    if sample_limit is not None:
        common = common[:sample_limit]
    n_clean = build_standardized_mode(clean_root, label_root, work_root / "clean", common, prefer_symlink)
    n_adv = build_standardized_mode(adv_root, label_root, work_root / "adv", common, prefer_symlink)
    n_den = build_standardized_mode(den_root, label_root, work_root / "denoised", common, prefer_symlink)
    n = min(n_clean, n_adv, n_den)
    save_meta(work_root, expected_meta, n)
    return work_root, n, False

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

def extract_yolo_val_metrics(results) -> EvalMetrics:
    md = getattr(results, "results_dict", {}) or {}
    return EvalMetrics(
        mAP50=finite_or_nan(md.get("metrics/mAP50(B)", md.get("metrics/mAP50", float("nan")))),
        mAP50_95=finite_or_nan(md.get("metrics/mAP50-95(B)", md.get("metrics/mAP50-95", float("nan")))),
        Precision=finite_or_nan(md.get("metrics/precision(B)", md.get("metrics/precision", float("nan")))),
        Recall=finite_or_nan(md.get("metrics/recall(B)", md.get("metrics/recall", float("nan")))),
        FPS=float("nan"),
        num_images_eval=0,
    )

def evaluate_mode_with_yolo(model, dataset_mode_root: Path, names_obj, device_spec, imgsz: int, conf: float, iou: float, verbose: bool = False) -> EvalMetrics:
    img_dir = dataset_mode_root / "images"
    image_files = list_images(img_dir)
    n = len(image_files)
    if n == 0:
        return EvalMetrics(num_images_eval=0)

    yaml_path = dataset_mode_root / "temp_eval.yaml"
    yaml_text = make_yaml_text(dataset_mode_root, names_obj)
    yaml_path.write_text(yaml_text, encoding="utf-8")

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
                results = model.val(
                    data=str(yaml_path),
                    device=device_spec,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    save_json=False,
                    plots=False,
                    verbose=verbose,
                )
                end_event.record()
                torch.cuda.synchronize()
                elapsed_s = start_event.elapsed_time(end_event) / 1000.0
        except Exception:
            t0 = time.perf_counter()
            results = model.val(
                data=str(yaml_path),
                device=device_spec,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                save_json=False,
                plots=False,
                verbose=verbose,
            )
            elapsed_s = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        results = model.val(
            data=str(yaml_path),
            device=device_spec,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            save_json=False,
            plots=False,
            verbose=verbose,
        )
        elapsed_s = time.perf_counter() - t0

    metrics = extract_yolo_val_metrics(results)
    metrics.num_images_eval = n
    metrics.FPS = float(n / elapsed_s) if elapsed_s > 1e-9 else float("nan")

    try:
        yaml_path.unlink()
    except Exception:
        pass
    if Path("runs").exists():
        shutil.rmtree("runs", ignore_errors=True)
    return metrics

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
    cells.append(cell_rect(cid, 20, 20, 1400, 40, title, 'rounded=1;whiteSpace=wrap;html=1;fontStyle=1;fontSize=20;fillColor=#dae8fc;strokeColor=#6c8ebf;'))
    cid += 1
    x0, y0 = 20, 80
    colw = [170] + [110] * (len(headers) - 1)
    if len(headers) > 8:
        colw = [160] + [100] * (len(headers) - 1)
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
            if j == 0:
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
    max_val = max([v for v in values if np.isfinite(v)] + [1e-6])
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default="./eval_standardized_basic")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--sample-limit", type=int, default=None)
    ap.add_argument("--include-ssim", action="store_true")
    ap.add_argument("--skip-image-metrics", action="store_true")
    ap.add_argument("--prefer-copy", action="store_true")
    ap.add_argument("--config-json", type=str, default=None)
    ap.add_argument("--verbose-val", action="store_true")
    ap.add_argument("--force-rebuild", action="store_true", help="Ignore existing standardized datasets and rebuild them.")
    args = ap.parse_args()

    if not ULTRA_OK:
        raise RuntimeError("ultralytics is not installed")

    cfg = json.loads(json.dumps(DEFAULTS))
    if args.config_json:
        cfg = json.load(open(args.config_json, "r", encoding="utf-8"))

    out_dir = ensure_dir(Path(args.output_dir))
    work_dir = ensure_dir(out_dir / "standardized_eval")
    drawio_dir = ensure_dir(out_dir / "drawio_xml")

    print(f"[INFO] Output: {out_dir}")
    print("[INFO] Loading YOLO detector...")
    model = YOLO(cfg["yolo_weight"])
    names_obj = getattr(model, "names", None)

    all_rows = []
    for exp in cfg["experiments"]:
        exp_group = exp["experiment_group"]
        attack = exp["attack"]
        clean_root = Path(cfg["clean_root"])
        adv_root = Path(exp["adv_root"])
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

            mode_rows = {}
            for mode_name, role_name in [("clean", "clean"), ("adv", "adv"), ("denoised", role)]:
                mode_root = triplet_root / mode_name
                print(f"[INFO] Official val() -> {exp_key} / {mode_name}")
                eval_metrics = evaluate_mode_with_yolo(
                    model=model,
                    dataset_mode_root=mode_root,
                    names_obj=names_obj,
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
        "experiment_group", "attack", "image_set", "role", "path",
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
    table_headers = ["experiment_group", "image_set", "mAP50", "mAP50_95", "Precision", "Recall", "FPS", "MSE", "MAE", "PSNR"]
    table_rows = []
    for r in all_rows:
        table_rows.append([str(r[h]) if h in {"experiment_group", "image_set"} else fmt(r[h]) for h in table_headers])
    doc.add_page("Summary_Table", build_table_page("FlowPure Standardized Evaluation", table_headers, table_rows))
    groups = sorted(set(r["experiment_group"] for r in all_rows))
    for grp in groups:
        grp_rows = [r for r in all_rows if r["experiment_group"] == grp and r["image_set"] != "Clean"]
        labels = [r["image_set"] for r in grp_rows]
        if labels:
            doc.add_page(f"{grp}_mAP50", build_bar_page(f"{grp} mAP50", labels, [r["mAP50"] for r in grp_rows], "mAP50"))
            doc.add_page(f"{grp}_Precision", build_bar_page(f"{grp} Precision", labels, [r["Precision"] for r in grp_rows], "Precision"))
            doc.add_page(f"{grp}_Recall", build_bar_page(f"{grp} Recall", labels, [r["Recall"] for r in grp_rows], "Recall"))
            doc.add_page(f"{grp}_PSNR", build_bar_page(f"{grp} PSNR", labels, [r["PSNR"] for r in grp_rows], "PSNR"))
    drawio_path = drawio_dir / "flowpure_standardized_eval.drawio.xml"
    doc.save(drawio_path)

    print(f"\n[INFO] Saved CSV: {csv_path}")
    print(f"[INFO] Saved JSON: {json_path}")
    print(f"[INFO] Saved draw.io XML: {drawio_path}")
    print("[INFO] Done.")

if __name__ == "__main__":
    main()

# python evaluate_flowpure_standardized_val_drawio_reuse.py \
#   --output-dir ./eval_standardized_basic \
#   --device cuda:0 \
#   --imgsz 640