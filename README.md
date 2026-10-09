# SAFE: Saliency-Guided Conditional Flow Matching for Adversarial Purification

This repository contains the four Python scripts released with the SAFE paper. SAFE uses an adversarial sensitivity saliency map (ASSM) to guide conditional flow matching for adversarial image purification before object detection.

## Repository status

This is an initial code release. **部分文件仍在整理中，后续会持续开源；Some files are still being organized and will be released progressively.** The current scripts are provided for research use and may require adaptation to your dataset layout, detector checkpoints, and hardware environment.

## Contents

- `gen_coco_adv_asm_AA.py` — generate APGD adversarial examples and ASM maps for COCO-style data.
- `fm_asm_aa_purifier_trainer_Perceptual_Loss.py` — train/test the ASSM-conditioned flow-matching purifier with perceptual loss.
- `evaluate_flowpure_standardized_val_drawio_reuse.py` — run standardized single-experiment detector and image-quality evaluation.
- `evaluate_flowpure_multi_reuse_drawio_no_intersection.py` — run multi-experiment evaluation and export CSV/JSON/draw.io summaries.

The committed experiment data are the tabular/JSON result summaries in the repository root and in `eval_ablation_three/`, `eval_flowpure_multi/`, and `eval_standardized_basic/`. Figures, `.drawio` sources, generated images, checkpoints, and raw datasets are intentionally excluded. Machine-specific paths in the released summaries are represented by `<FLOWPURE_ROOT>`.

## Environment

The scripts are intended for a CUDA-enabled Python environment. Install the packages needed by the workflow you use; a starting point is:

```bash
pip install numpy pillow pyyaml scipy scikit-image opencv-python tqdm \
  torch torchvision torchcfm ultralytics torchattacks pycocotools
```

Match the PyTorch/torchvision build to your CUDA version. Some evaluation backends are optional and are detected at runtime.

## Data and checkpoints

The released scripts use COCO-style image/label directories and detector or purifier checkpoints. Several default configuration entries retain the paths from the authors' training environment (for example, `/root/wangchao/FlowPure/...`). Before running, update those entries through the script configuration or `--config-json` so they point to your local dataset and weights. Datasets, model weights, generated images, and experiment outputs are not included in this repository.

## Typical usage

Generate adversarial examples and saliency maps:

```bash
python gen_coco_adv_asm_AA.py
```

Train the purifier (all data directories are required):

```bash
python fm_asm_aa_purifier_trainer_Perceptual_Loss.py \
  --mode train \
  --adv_dir /path/to/adv_images \
  --sal_dir /path/to/asm_npy_maps \
  --clean_dir /path/to/coco \
  --output_dir ./fm_vgg_out
```

Run standardized evaluation:

```bash
python evaluate_flowpure_standardized_val_drawio_reuse.py \
  --config-json /path/to/config.json \
  --output-dir ./eval_standardized_basic
```

For multi-experiment evaluation, use the corresponding `evaluate_flowpure_multi_reuse_drawio_no_intersection.py` entry point and provide a compatible configuration. Run `--help` for all available options.

## Reproducibility notes

Results depend on the exact COCO split, attack settings, detector checkpoints, purifier checkpoints, CUDA/PyTorch versions, and configuration paths. Please record these settings when reproducing the paper results. The scripts may download pretrained weights through their upstream libraries when a checkpoint is not supplied.

## Citation

If you use this code, please cite the accompanying paper:

```text
SAFE: Saliency-Guided Conditional Flow Matching for Adversarial Purification.
```

## License

The source code is released under the MIT License; see `LICENSE`.
