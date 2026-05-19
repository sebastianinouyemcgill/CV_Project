# CV Project — NYU Depth Estimation

Monocular depth estimation with a **pretrained ResNet50-UNet** on NYU Depth V2.

## Quick start

```bash
export NYU_DATA_ROOT="/path/to/CV_Project/data"
pip install -r requirements.txt
python smoke_train.py          # sanity check
python train.py                  # full training
python train.py --resume         # resume latest run (checkpoints/runN/last.pt)
python evaluate.py --run-name run1
tensorboard --logdir checkpoints/run1/tensorboard
```

## Architecture

- **Default:** `ResNetEncoderUNet` with ImageNet-pretrained ResNet50 encoder + 4-scale UNet decoder
- **Baseline:** `UNet` (tiny, no pretrain) via `MODEL_NAME=unet`
- Output: sigmoid → normalized depth in `[0, 1]` (×10 m for metrics)

## Loss

Combined depth loss (`losses.py`):
- Scale-invariant log loss (primary)
- BerHu (robust regression)
- Gradient L1 (edge preservation)
- Edge-aware smoothness (optional, weight 0.01)

## Training defaults

| Setting | Default |
|---------|---------|
| Optimizer | AdamW, lr=1e-4, wd=1e-2 |
| Scheduler | Warmup (2 epochs) + cosine decay |
| AMP | On (CUDA only) |
| Grad clip | 1.0 |
| Train subset | 8000 random frames |
| Epochs | 20 |

## Environment variables

| Variable | Meaning |
|----------|---------|
| `NYU_DATA_ROOT` | Dataset folder (`data/` or project root) |
| `CHECKPOINT_DIR` | Model checkpoints |
| `MODEL_NAME` | `resnet50` / `resnet34` / `unet` |
| `PRETRAINED` | `1` or `0` |
| `MAX_TRAIN_SAMPLES` | Subset size (`0` = full 50k) |
| `USE_FULL_TRAIN` | `1` to use all train data |
| `RUN_NAME` | Force folder name (`run1`, `my_exp`); empty = auto `runN` |
| `SAVE_EVERY_EPOCH` | Save `epochs/epoch_XXX.pt` each epoch (`1` default) |

## Experiment folders

Each training run writes to `CHECKPOINT_DIR/run1/`, `run2/`, …:

- `best.pt` — best validation RMSE
- `last.pt` — resume training
- `epochs/epoch_001.pt` — checkpoint every epoch
- `metrics_curves.png` — loss / RMSE / δ₁ plots (updated each epoch)
- `metrics_history.json` — full metric log
- `eval_result.png` — from `evaluate.py`

## Colab

See `colab/train_nyu_depth.ipynb` — stage data to `/content/nyu_data`, train, evaluate.

## Target metrics (NYU indoor)

| Metric | Strong target |
|--------|---------------|
| δ₁ | > 0.80 |
| RMSE | < 0.6 m |

After these improvements, expect δ₁ ~0.55–0.70 on 8k subset / 20 epochs; full 50k training needed for δ₁ > 0.80.
