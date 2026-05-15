# NegMerge Reproducibility Study

**COMP6258 Reproducibility Challenge — University of Southampton**

This repository contains the reproduction of key experiments from:

> **NegMerge: Consensual Weight Negation for Strong Machine Unlearning**
> Park et al., ICML 2025

**GitHub Repository:** https://github.com/Abdu119/NegMerge-Reproducibility-Study

## Overview

This study reproduces NegMerge's machine unlearning experiments using CLIP ViT-B/32 with Stanford Cars as the forget set and ImageNet as the retain set. The reproduction was conducted on an Apple M3 Max (MPS backend) rather than CUDA GPUs.

In addition to verifying the paper's three numerical claims, we extend the study with an **N-variation ablation** that the original authors did not run. This sweeps the number of fine-tuned models merged from N=3 up to N=30 (the paper's fixed value) and exposes a finding the paper missed: forget quality is **non-monotonic** in N, because the optimal coefficient α shifts in discrete steps as N grows.

### Key Results

| Metric | Paper | Our Reproduction |
|--------|-------|------------------|
| Cars accuracy (forget) | ~27–28% | **24.08%** |
| ImageNet accuracy (retain) | ~60–61% | **59.27%** |
| Sign consensus rate (N=30) | ~10% | **9.66%** |
| Optimal coefficient α | — | **0.95** |
| Total compute | — | **4.2 hours** on Apple Silicon GPU via MPS |

All three of the paper's central claims were **confirmed**.

### N-Variation Finding (Our Extension)

We swept N ∈ {3, 5, 10, 15, 20, 25, 30} with three random subsets per N. Two findings stand out:

1. **N = 20 already matches N = 30** in forget quality (24.04% vs 24.08% Cars).
2. **At matched α = 0.90, N = 20 outperforms N = 30** (24.04% vs 26.41% Cars). So N=30's advantage in the paper comes from tolerating a higher α (0.95), not from a cleaner merged signal.

See `report/paper.pdf` for the full analysis and statistical evidence (Cohen's d effect-size framing across matched subsets).

## Repository Structure

```
.
├── NegMerge.ipynb            # Main reproduction notebook ⭐
├── analysis_n_variation.py   # N-variation ablation script ⭐
├── results/                  # Stored ablation outputs (csv/json/log)
├── report/
│   ├── paper.pdf             # Reproducibility report ⭐
│   └── paper.tex             # LaTeX source
├── README.md
├── LICENSE
├── NOTICE
├── environment.yml
├── .gitignore
├── src/                      # Source code modules
│   ├── eval.py               # Evaluation functions
│   ├── modeling.py           # Model definitions
│   ├── task_vectors.py       # Task vector implementation
│   ├── utils.py              # Utilities (includes MaybeToTensor fix)
│   ├── heads.py              # Classification head utilities
│   ├── finetune.py           # Fine-tuning code (not used in notebook)
│   ├── linearize.py          # Linearization utilities
│   ├── args.py               # Argument parsing
│   ├── distributed.py        # Distributed training utils
│   └── datasets/
│       ├── cars_hf.py        # Stanford Cars (HuggingFace) ⭐
│       ├── imagenet.py       # ImageNet loader
│       ├── imagenet_hf.py    # ImageNet fallback loaders
│       └── ...               # Other datasets
├── checkpoints/              # Model weights (NOT in git — ~90GB)
│   └── standard/ViT-B-32/
│       ├── zeroshot.pt
│       ├── zeroshot_accuracies.json
│       ├── head_CarsVal.pt
│       ├── head_ImageNetVal.pt
│       └── checkpoints_rand-m{1-10}-n{1-3}/
└── dataset/                  # Dataset files (NOT in git)
    ├── imagenet/val/
    └── stanford_cars/        # Auto-downloaded from HuggingFace
```

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone https://github.com/Abdu119/NegMerge-Reproducibility-Study.git
cd NegMerge-Reproducibility-Study
conda env create -f environment.yml
conda activate negmerge
```

### 2. Download Checkpoints (~90GB)

Download from [Google Drive](https://drive.google.com/drive/u/1/folders/1m1iHi5KoTN1Fg5JqIZxtVP1ZTxgILZyi) and extract to `checkpoints/standard/ViT-B-32/`.

Required files:
- `zeroshot.pt` (pretrained CLIP)
- `zeroshot_accuracies.json`
- `head_CarsVal.pt`, `head_ImageNetVal.pt`
- 30 fine-tuned checkpoints: `checkpoints_rand-m{1-10}-n{1-3}/CarsVal/finetuned.pt`

### 3. Prepare Datasets

**Stanford Cars**: Automatically downloaded from HuggingFace (`tanganke/stanford_cars`) on first run.

**ImageNet**: Place the face-blurred validation set in `dataset/imagenet/val/` with class subfolders. Note that the paper does not specify which ImageNet variant it uses; we measured a 62.37% baseline on the face-blurred variant vs 66.66% reported for standard ImageNet — a 4pp difference that propagates into the 95% retention threshold.

### 4. Run the Main Reproduction

```bash
jupyter notebook NegMerge.ipynb
```

Runtime: ~4 hours on M3 Max.

### 5. Run the N-Variation Ablation (Optional)

```bash
python analysis_n_variation.py
```

Outputs land in `results/`.

## Code Adaptations

Several modifications were required to run the authors' code on MPS hardware and with currently-available data sources.

### 1. MaybeToTensor Fix

PIL images passed through `timm`'s `MaybeToTensor` unconverted in the MPS pipeline, silently producing incorrect outputs rather than raising an error. We patched it to explicitly convert PIL inputs to tensors:

```python
class MaybeToTensor:
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return to_tensor(x)
```

### 2. Stanford Cars from HuggingFace

The original Stanford URLs return 404 errors. We source the dataset from `tanganke/stanford_cars` on HuggingFace and verify correctness by matching the expected pretrained Cars accuracy (59.58%).

### 3. Threshold Recalculation

Our ImageNet baseline is 62.37% (face-blurred variant) vs the 66.66% stored in the original codebase. We recalculated the 95% retention threshold accordingly: 0.95 × 62.37% = 59.25%.

### 4. PyTorch 2.6 Compatibility

Set `weights_only=False` in `torch.load()` for the stricter security defaults in PyTorch 2.6.

### 5. ViT Patching

Applied `patch_vision_transformer_deep()` to all loaded models for architecture consistency.

## Hardware Requirements

- **Tested on**: Apple M3 Max, 36GB unified memory
- **Also compatible with**: CUDA GPUs (the original target)
- **Storage**: ~90GB for checkpoints, ~150GB for ImageNet
- **RAM**: 32GB+ recommended

Compute used the M3 Max's integrated GPU via Apple's MPS backend — no discrete (NVIDIA/CUDA) GPU required. Total time was 4.2 hours across 19 (N, seed) configurations.

## Citation

If you use this reproduction or the N-variation ablation, please cite the original paper:

```bibtex
@inproceedings{park2025negmerge,
  title={NegMerge: Consensual Weight Negation for Strong Machine Unlearning},
  author={Park, Jaehan and Lee, Jongwoo and Ham, Junghoon and Choi, Minjoon},
  booktitle={ICML},
  year={2025}
}
```

## Acknowledgments

- Original NegMerge authors for releasing code and checkpoints
- COMP6258 Deep Learning module, University of Southampton
- Built on top of [Tangent Arithmetic](https://github.com/gortizji/tangent_task_arithmetic)

## License

This reproduction follows the original repository's MIT license. See [LICENSE](LICENSE).
