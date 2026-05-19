import os
from pathlib import Path

# Folder that directly contains nyu2_train.csv, nyu2_test.csv, nyu2_train/, nyu2_test/.
_DEFAULT_DATA_ROOT = "/Volumes/SSD 2/Projects/CV Project/data"

CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
LOG_DIR = Path(os.environ.get("LOG_DIR", "runs"))
VIZ_DIR = Path(os.environ.get("VIZ_DIR", "visualizations"))


def resolve_data_root(path: Path) -> Path:
    path = Path(path)
    if (path / "nyu2_train.csv").is_file():
        return path
    data_sub = path / "data"
    if (data_sub / "nyu2_train.csv").is_file():
        return data_sub
    return path


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


DATA_ROOT = resolve_data_root(Path(os.environ.get("NYU_DATA_ROOT", _DEFAULT_DATA_ROOT)))
TRAIN_CSV = DATA_ROOT / "nyu2_train.csv"
TEST_CSV = DATA_ROOT / "nyu2_test.csv"

# NYU Depth V2 indoor range
MAX_DEPTH_M = 10.0
MIN_DEPTH_M = 1e-3

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 2
GRAD_ACCUM_STEPS = 1
GRAD_CLIP_NORM = 1.0
WARMUP_EPOCHS = 2

# Model
MODEL_NAME = os.environ.get("MODEL_NAME", "resnet50")  # unet | resnet34 | resnet50
PRETRAINED = _env_bool("PRETRAINED", True)

# Training
USE_AMP = _env_bool("USE_AMP", True)
USE_COMPILE = _env_bool("USE_COMPILE", False)
PERSISTENT_WORKERS = _env_bool("PERSISTENT_WORKERS", True)
VIZ_EVERY = int(os.environ.get("VIZ_EVERY", "5"))

# ImageNet normalization (required for pretrained encoders)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Loss weights
LOSS_SIL_WEIGHT = 1.0
LOSS_BERHU_WEIGHT = 0.5
LOSS_GRAD_WEIGHT = 0.1
LOSS_SMOOTH_WEIGHT = 0.01

# Train subset
def _parse_max_train_samples():
    if os.environ.get("USE_FULL_TRAIN", "0") == "1":
        return None
    raw = os.environ.get("MAX_TRAIN_SAMPLES", "8000")
    if raw in ("0", "", "none", "None", "full"):
        return None
    return int(raw)


MAX_TRAIN_SAMPLES = _parse_max_train_samples()
TRAIN_SUBSAMPLE_SEED = 42

SEED = int(os.environ.get("SEED", "42"))

# Experiment run: checkpoints/run1/, run2/, ... (set RUN_NAME to force a folder name)
RUN_NAME = os.environ.get("RUN_NAME", "").strip()
SAVE_EVERY_EPOCH = _env_bool("SAVE_EVERY_EPOCH", True)
