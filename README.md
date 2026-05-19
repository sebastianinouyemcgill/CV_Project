# CV Project — NYU Depth Estimation

Monocular depth estimation with a U-Net on NYU Depth V2 (`nyu2_train` / `nyu2_test` + CSV splits).

## Dataset layout

`NYU_DATA_ROOT` must be the folder that **directly contains** the four items:

```
$NYU_DATA_ROOT/
  nyu2_train.csv
  nyu2_test.csv
  nyu2_train/
  nyu2_test/
```

You can set `NYU_DATA_ROOT` to either that folder or its parent (e.g. `CV_Project` or `CV_Project/data` on Drive).

CSV rows use a `data/` prefix (`data/nyu2_train/...`); the loader strips it and joins under `NYU_DATA_ROOT`.

## Local training (Mac / SSD)

```bash
export NYU_DATA_ROOT="/Volumes/SSD 2/Projects/CV Project/data"   # or project root
export CHECKPOINT_DIR="./checkpoints"                             # optional
python smoke_train.py   # quick check
python train.py         # full run
```

## Colab + Google Drive (recommended for full training)

The dataset is ~4 GB but ~200k tiny files. **Unzipping on Drive is very slow.** Use this workflow:

1. **On your Mac** — create one archive (once):
   ```bash
   cd "/path/to/folder/with/csvs"
   tar -czf nyu_dataset.tar.gz nyu2_train.csv nyu2_test.csv nyu2_train nyu2_test
   ```
2. **Upload** `nyu_dataset.tar.gz` to `My Drive/CV_Project/data/` (Drive for desktop is fine).
3. **Each Colab session** — open `colab/train_nyu_depth.ipynb`, mount Drive, extract the tar to `/content/nyu_data` (fast local disk), train, save checkpoints to `CV_Project/checkpoints/`.

If you already synced the extracted folders to Drive (no tar), set `MODE = "rsync"` in the notebook to copy `CV_Project/data/` → `/content/nyu_data` once per session.

| Location | Role |
|----------|------|
| `CV_Project/data/` on Drive | Long-term storage (tar and/or extracted files) |
| `/content/nyu_data` in Colab | Fast scratch copy for training |
| `CV_Project/checkpoints/` on Drive | Saved models (`best_unet.pt`) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Config

Edit `config.py` or environment variables:

| Variable | Meaning |
|----------|---------|
| `NYU_DATA_ROOT` | Path to dataset folder (see layout above) |
| `CHECKPOINT_DIR` | Where `best_unet.pt` is saved |
