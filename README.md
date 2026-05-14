# ATLAS — Automated Temporal Land-change Analysis System

> Binary change detection on co-registered pre/post-event satellite imagery (EO + SAR).  
> Built for the GalaxEye technical assignment.

**ATLAS** is a complete, reproducible pipeline that ingests pairs of satellite images taken before and after an event, and produces pixel-level maps of what changed — with uncertainty estimates, test-time augmentation, and an interactive results dashboard.

---

## What This Project Does

Given two satellite images of the same location taken at different times (before and after an event like a disaster), the model identifies which pixels have changed — specifically detecting **building damage**. Each pixel is classified as either:

- **0 — No Change** (background, intact buildings)
- **1 — Change** (damaged or destroyed buildings)

---

## Why We Built It This Way

| Design Choice | Reason |
|---|---|
| **UNet architecture** | Classic encoder-decoder with skip connections — well-suited to pixel-level segmentation tasks. Lightweight enough to train fast without a large GPU. |
| **Stacked pre+post channels as input** | Instead of two separate networks, pre and post images are concatenated channel-wise and fed to one network. This forces the model to learn the difference itself. |
| **BCEWithLogitsLoss with `pos_weight=5`** | Change pixels are rare (class imbalance). Upweighting positives stops the model from lazily predicting "no change" everywhere. |
| **Weighted sampler (`change_weight_multiplier=5`)** | Patches that contain change pixels are drawn more often during training. More exposure to the rare class = better learning. |
| **Label remapping to binary** | The raw masks have 4 classes (Background, Intact, Damaged, Destroyed). We collapse them: Damaged+Destroyed → 1 (Change), Background+Intact → 0 (No Change). |
| **Cosine Annealing LR scheduler** | Helps the model escape local minima and converge more smoothly across epochs. |
| **Threshold sweep on val set** | The default sigmoid threshold of 0.5 is rarely optimal. After training, we sweep thresholds from 0.05 to 0.95 and pick the one with the best F1 on the validation set. This threshold is saved into the checkpoint and used automatically at eval time. |
| **8-fold Test-Time Augmentation (TTA)** | At inference, the image is fed through the model in all 8 orientations of the dihedral group D4 (identity, flips, rotations). Predictions are averaged back in the original orientation. This consistently improves F1 and IoU without any retraining. |
| **MC Dropout uncertainty** | Dropout layers stay active at inference for 30 stochastic forward passes. The standard deviation across passes gives a per-pixel uncertainty map — highlights where the model is unsure. |
| **Streamlit dashboard** | Interactive UI for inspecting training curves, threshold sweep, metrics, qualitative examples, and live inference without writing any code. |

---

## Project Structure

```
Change_dataset assignment/
├── data/
│   ├── train/
│   │   ├── pre-event/        ← pre-event satellite images (.tif)
│   │   ├── post-event/       ← post-event satellite images (.tif)
│   │   └── target/           ← ground-truth masks (0–3 labels)
│   ├── val/
│   │   └── (same structure)
│   └── test/
│       └── (same structure, no target/ needed)
│
├── code/
│   ├── configs/
│   │   └── config.yaml       ← all hyperparameters and paths
│   ├── train.py              ← training script
│   ├── eval.py               ← evaluation + qualitative visualisation
│   ├── dashboard.py          ← Streamlit results dashboard
│   ├── inspect_sample_shapes.py ← debug utility
│   └── galaxeye_cd/          ← core library
│       ├── model.py          ← UNetSmall architecture
│       ├── dataset.py        ← data loading, augmentation, layout discovery
│       ├── metrics.py        ← IoU, F1, precision, recall, threshold sweep
│       ├── sampler.py        ← weighted sampler for class imbalance
│       ├── tta.py            ← 8-fold D4 test-time augmentation
│       ├── analysis.py       ← MC Dropout + qualitative visualisation
│       ├── config.py         ← config dataclass + YAML loader
│       └── utils.py          ← seed, device, JSON/YAML helpers
│
├── runs/                     ← auto-created by train.py
│   └── baseline_unet/
│       ├── checkpoints/
│       │   ├── best.pth      ← best model by val F1
│       │   └── epoch_N.pth
│       ├── training_history.json
│       ├── metrics_val.json
│       ├── metrics_test.json
│       ├── threshold_sweep.json
│       ├── config_resolved.yaml
│       └── qualitative/
│           ├── val/          ← PNG grids from eval on val
│           └── test/         ← PNG grids from eval on test
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

**Dependencies installed:**

| Package | Version | Purpose |
|---|---|---|
| torch | 2.4.1 | Deep learning framework |
| torchvision | 0.19.1 | Image utilities |
| numpy | 2.1.1 | Array operations |
| opencv-python | 4.10.0.84 | Image I/O and resizing |
| tqdm | 4.66.5 | Progress bars |
| PyYAML | 6.0.2 | Config file parsing |
| scikit-learn | 1.5.2 | Utilities |
| matplotlib | 3.9.2 | Qualitative figure generation |
| plotly | 5.24.1 | Interactive dashboard charts |
| streamlit | 1.38.0 | Results dashboard |
| Pillow | 10.4.0 | Image display in dashboard |
| pandas | 2.2.3 | DataFrame for charts |

---

## Dataset Layout

Place data under `data/` as follows:

```
data/
├── train/
│   ├── pre-event/    ← filenames must match across folders
│   ├── post-event/
│   └── target/       ← ground-truth masks (4-class: 0,1,2,3)
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    └── post-event/   ← no target/ required for inference-only
```

The dataset loader **auto-discovers** this layout. No manual path wiring needed.

### Label Remapping

Original 4-class masks are remapped to binary automatically:

| Original Label | Meaning | Binary |
|---|---|---|
| 0 | Background | 0 (No Change) |
| 1 | Intact | 0 (No Change) |
| 2 | Damaged | **1 (Change)** |
| 3 | Destroyed | **1 (Change)** |

---

## How to Run Everything

### Step 1 — Train

```powershell
python code/train.py --config code/configs/config.yaml
```

What happens:
- Loads train and val splits
- Builds weighted sampler (oversamples change-heavy patches)
- Trains `UNetSmall` with BCEWithLogitsLoss + AdamW + Cosine LR scheduler
- Saves `best.pth` checkpoint whenever val F1 improves
- Saves `epoch_N.pth` every epoch
- After training: runs threshold sweep on val set and stores best threshold in `best.pth`
- Writes `runs/baseline_unet/training_history.json`

Console output per epoch:
```
[epoch 001] loss=0.3421  val_iou=0.4123  val_f1=0.5831  val_p=0.6102  val_r=0.5594
```

---

### Step 2 — Evaluate on Validation Set

```powershell
python code/eval.py --config code/configs/config.yaml --split val --weights runs/baseline_unet/checkpoints/best.pth
```

---

### Step 3 — Evaluate on Test Set

```powershell
python code/eval.py --config code/configs/config.yaml --split test --weights runs/baseline_unet/checkpoints/best.pth
```

Console output:
```
--------------------------------------------------
  Split     : test
  IoU       : 0.5234
  F1        : 0.6873
  Precision : 0.7102
  Recall    : 0.6659
  Confusion matrix (gt rows x pred cols):
    [TP_count, FP_count]
    [FN_count, TN_count]
  Saved -> runs/baseline_unet/metrics_test.json
```

Also saves qualitative PNGs to `runs/baseline_unet/qualitative/test/`.

#### Optional eval flags

| Flag | Effect |
|---|---|
| `--no-tta` | Disable test-time augmentation |
| `--no-mc` | Disable MC Dropout uncertainty |
| `--no-vis` | Skip saving qualitative PNG visualisations |
| `--out path/to/file.json` | Write metrics to a custom path |

---

### Step 4 — Launch the Dashboard

```powershell
streamlit run code/dashboard.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

Dashboard tabs:

| Tab | What it shows |
|---|---|
| **Overview** | Best F1, best IoU, decision threshold, full config |
| **Training Curves** | Interactive plots: loss, F1, IoU, LR, precision, recall per epoch |
| **Threshold Sweep** | F1 / precision / recall vs threshold curve, best threshold marked |
| **Results** | Val + test metrics (IoU, F1, precision, recall) + confusion matrix heatmap |
| **Gallery** | Qualitative PNG examples saved by eval.py |
| **Live Inference** | Pick any sample, run model on the spot, see pre/post/prediction/error/uncertainty |

---

## Configuration Reference (`code/configs/config.yaml`)

```yaml
run_name: "baseline_unet"       # subfolder under runs/
seed: 42
device: "cuda"                   # "cuda" or "cpu"

data_root: "data"
img_size: 256                    # resize all patches to 256×256
batch_size: 4
num_workers: 2

splits:
  train: "data/train"
  val:   "data/val"
  test:  "data/test"

# Leave null = auto-discover folders inside each split
folders:
  eo_pre:     null
  eo_post:    null
  sar_pre:    null
  sar_post:   null
  pre_event:  null
  post_event: null
  target:     null

dropout_p: 0.3                   # MC Dropout probability

epochs: 10
lr: 0.0003
weight_decay: 0.00001

lr_scheduler: "cosine"           # CosineAnnealingWarmRestarts
lr_T0: 10                        # restart every 10 epochs
lr_T_mult: 2                     # period doubles: 10, 20, 40...
lr_eta_min: 0.00001

augment: true                    # train-only geometric + photometric augmentation
pos_weight: 5.0                  # upweight change class in loss
change_weight_multiplier: 5.0    # weighted sampler multiplier

threshold_sweep: true            # sweep val thresholds after training
save_every: 1                    # save checkpoint every N epochs
metric_for_best: "f1"            # track best checkpoint by F1

tta: true                        # 8-fold D4 TTA at eval
mc_dropout_passes: 30            # MC Dropout passes (0 = disabled)
num_vis_samples: 10              # qualitative PNG examples to save
```

---

## Model Architecture — UNetSmall

```
Input (B, C, 256, 256)         C = 2× channels per image (pre+post stacked)
    │
    ├─ Encoder
    │     enc1: ConvBlock(C → 32)   + MaxPool
    │     enc2: ConvBlock(32 → 64)  + MaxPool
    │     enc3: ConvBlock(64 → 128) + MaxPool
    │
    ├─ Bottleneck
    │     ConvBlock(128 → 256) + Dropout2d(0.3)
    │
    └─ Decoder (with skip connections from encoder)
          dec3: ConvTranspose + ConvBlock(256+128 → 128) + Dropout2d(0.15)
          dec2: ConvTranspose + ConvBlock(128+64  → 64)  + Dropout2d(0.15)
          dec1: ConvTranspose + ConvBlock(64+32   → 32)
          head: Conv1×1(32 → 1)   ← raw logit map

Output (B, 1, 256, 256)        apply sigmoid → probability, threshold → binary mask
```

Each `ConvBlock` = Conv3×3 → BatchNorm → ReLU → Conv3×3 → BatchNorm → ReLU.

For **MC Dropout**, call `model.enable_mc_dropout()` after `model.eval()` — this sets only Dropout layers back to train mode while keeping BatchNorm in eval mode.

---

## Outputs Summary

After a full train + eval run, `runs/baseline_unet/` contains:

```
runs/baseline_unet/
├── checkpoints/
│   ├── best.pth                ← best val F1 checkpoint + best_threshold baked in
│   └── epoch_1.pth, epoch_2.pth, ...
├── training_history.json       ← loss, F1, IoU, LR per epoch
├── metrics_val.json            ← val metrics from train.py
├── metrics_test.json           ← test metrics from eval.py
├── threshold_sweep.json        ← full sweep results + best threshold
├── config_resolved.yaml        ← exact config + env info used for this run
└── qualitative/
    ├── val/  *.png             ← 10 PNGs: pre | post | GT | pred | error | uncertainty
    └── test/ *.png
```

---

## Metrics Reported

All metrics are computed on the **change class (label = 1)** only:

| Metric | Formula |
|---|---|
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1** | 2 × Precision × Recall / (Precision + Recall) |
| **IoU** | TP / (TP + FP + FN) |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `batch_size` to 2 in `config.yaml` |
| `No images found in pre-event/` | Check filenames match exactly across `pre-event/`, `post-event/`, `target/` |
| `streamlit: command not found` | Run `pip install streamlit` or use `python -m streamlit run code/dashboard.py` |
| Dashboard shows "No runs found" | Train first with `train.py` — the `runs/` folder must exist |
| Eval gives all zeros | Verify `best.pth` path is correct and `--split` matches the split you have masks for |
