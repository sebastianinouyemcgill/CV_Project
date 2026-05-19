import os
from pathlib import Path

# Folder that directly contains nyu2_train.csv, nyu2_test.csv, nyu2_train/, nyu2_test/.
# Override with NYU_DATA_ROOT. You can pass either .../data or the project root (.../CV_Project).
_DEFAULT_DATA_ROOT = "/Volumes/SSD 2/Projects/CV Project/data"

# Where checkpoints are written. On Colab, set to your Drive checkpoints folder.
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))


def resolve_data_root(path: Path) -> Path:
    """Accept NYU_DATA_ROOT as either the data/ folder or its parent project folder."""
    path = Path(path)
    if (path / "nyu2_train.csv").is_file():
        return path
    data_sub = path / "data"
    if (data_sub / "nyu2_train.csv").is_file():
        return data_sub
    return path


DATA_ROOT = resolve_data_root(Path(os.environ.get("NYU_DATA_ROOT", _DEFAULT_DATA_ROOT)))

TRAIN_CSV = DATA_ROOT / "nyu2_train.csv"
TEST_CSV = DATA_ROOT / "nyu2_test.csv"

MAX_DEPTH_M = 10.0
IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
