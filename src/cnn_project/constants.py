from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_PATH: Path = PROJECT_ROOT / "DATA"
TRAIN_DATA_C: Path = DATA_PATH / "classification_task" / "train"
TEST_DATA_C: Path = DATA_PATH / "classification_task" / "test"
TRAIN_DATA_S: Path = DATA_PATH / "segmentation_task" / "train"
TEST_DATA_S: Path = DATA_PATH / "segmentation_task" / "test"

IMAGE_SIZE: tuple[int, int] = (150, 150)
BATCH_SIZE: int = 32
SEED: int = 42
