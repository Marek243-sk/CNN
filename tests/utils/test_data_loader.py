"""Tests for src/cnn_project/data_loader.py"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cnn_project.utils import data_loader

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
SMALL_SIZE = (8, 8)
SMALL_BATCH = 4


def _make_images(directory: Path, count: int, *, ext: str = ".jpg") -> None:
    """Write ``count`` tiny random RGB images into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(count):
        pixels = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(directory / f"img_{i}{ext}")


def _make_split(root: Path, classes: list[str], per_class: int = 3) -> Path:
    for name in classes:
        _make_images(root / name, per_class)
    return root


@pytest.fixture
def dataset_dirs(tmp_path: Path) -> tuple[Path, Path]:
    train = _make_split(tmp_path / "train", CLASSES)
    test = _make_split(tmp_path / "test", CLASSES)
    return train, test


@pytest.fixture(autouse=True)
def small_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests fast by shrinking image and batch size."""
    monkeypatch.setattr(data_loader, "IMAGE_SIZE", SMALL_SIZE)
    monkeypatch.setattr(data_loader, "BATCH_SIZE", SMALL_BATCH)


# --------------------------------------------------------------------------
# get_class_names
# --------------------------------------------------------------------------
class TestGetClassNames:
    def test_returns_sorted_names(self, tmp_path: Path) -> None:
        for name in ["pituitary", "glioma", "no_tumor"]:
            (tmp_path / name).mkdir()

        assert data_loader.get_class_names(tmp_path) == [
            "glioma",
            "no_tumor",
            "pituitary",
        ]

    def test_ignores_files_and_hidden_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "glioma").mkdir()
        (tmp_path / ".ipynb_checkpoints").mkdir()
        (tmp_path / "notes.txt").write_text("hi")

        assert data_loader.get_class_names(tmp_path) == ["glioma"]

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            data_loader.get_class_names(tmp_path / "does_not_exist")

    def test_path_is_a_file_raises(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file.txt"
        file_path.write_text("x")

        with pytest.raises(FileNotFoundError):
            data_loader.get_class_names(file_path)

    def test_no_subfolders_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No subfolders"):
            data_loader.get_class_names(tmp_path)


# --------------------------------------------------------------------------
# validate_dataset_structure
# --------------------------------------------------------------------------
class TestValidateDatasetStructure:
    def test_valid_structure_returns_class_names(
        self, dataset_dirs: tuple[Path, Path]
    ) -> None:
        train, test = dataset_dirs

        assert data_loader.validate_dataset_structure(train, test) == CLASSES

    def test_mismatched_classes_raise(self, tmp_path: Path) -> None:
        train = _make_split(tmp_path / "train", CLASSES)
        test = _make_split(tmp_path / "test", CLASSES[:-1])

        with pytest.raises(ValueError, match="does not match"):
            data_loader.validate_dataset_structure(train, test)

    def test_empty_class_folder_raises(self, tmp_path: Path) -> None:
        train = _make_split(tmp_path / "train", CLASSES)
        test = _make_split(tmp_path / "test", CLASSES[:-1])
        (test / CLASSES[-1]).mkdir()  # same class name, but no images

        with pytest.raises(ValueError, match="no images"):
            data_loader.validate_dataset_structure(train, test)

    def test_non_image_files_do_not_count(self, tmp_path: Path) -> None:
        train = _make_split(tmp_path / "train", CLASSES[:1])
        test = tmp_path / "test"
        (test / CLASSES[0]).mkdir(parents=True)
        (test / CLASSES[0] / "readme.txt").write_text("not an image")

        with pytest.raises(ValueError, match="no images"):
            data_loader.validate_dataset_structure(train, test)

    def test_uppercase_extension_is_accepted(self, tmp_path: Path) -> None:
        train = tmp_path / "train"
        test = tmp_path / "test"
        _make_images(train / "glioma", 1, ext=".PNG")
        _make_images(test / "glioma", 1, ext=".PNG")

        assert data_loader.validate_dataset_structure(train, test) == ["glioma"]

    def test_missing_split_raises(self, tmp_path: Path) -> None:
        train = _make_split(tmp_path / "train", CLASSES)

        with pytest.raises(FileNotFoundError):
            data_loader.validate_dataset_structure(train, tmp_path / "missing")


# --------------------------------------------------------------------------
# load_datasets
# --------------------------------------------------------------------------
class TestLoadDatasets:
    def test_returns_datasets_and_class_names(
        self, dataset_dirs: tuple[Path, Path]
    ) -> None:
        train_ds, test_ds, class_names = data_loader.load_datasets(*dataset_dirs)

        assert class_names == CLASSES
        assert train_ds is not None
        assert test_ds is not None

    def test_batch_shapes_and_dtypes(
        self, dataset_dirs: tuple[Path, Path]
    ) -> None:
        train_ds, _, _ = data_loader.load_datasets(*dataset_dirs)

        images, labels = next(iter(train_ds))

        assert images.shape[1:] == (*SMALL_SIZE, 3)
        assert images.shape[0] <= SMALL_BATCH
        assert labels.dtype.is_integer
        assert labels.shape[0] == images.shape[0]

    def test_all_samples_are_loaded(
        self, dataset_dirs: tuple[Path, Path]
    ) -> None:
        train_ds, test_ds, _ = data_loader.load_datasets(*dataset_dirs)

        expected = len(CLASSES) * 3
        assert sum(int(x.shape[0]) for x, _ in train_ds) == expected
        assert sum(int(x.shape[0]) for x, _ in test_ds) == expected

    def test_labels_match_class_name_index(self, tmp_path: Path) -> None:
        train = _make_split(tmp_path / "train", CLASSES, per_class=2)
        test = _make_split(tmp_path / "test", CLASSES, per_class=2)

        _, test_ds, class_names = data_loader.load_datasets(train, test)

        labels = np.concatenate([y.numpy() for _, y in test_ds])
        assert set(labels.tolist()) == set(range(len(class_names)))

    def test_test_split_is_not_shuffled(
        self, dataset_dirs: tuple[Path, Path]
    ) -> None:
        _, test_ds, _ = data_loader.load_datasets(*dataset_dirs)

        first = np.concatenate([y.numpy() for _, y in test_ds])
        second = np.concatenate([y.numpy() for _, y in test_ds])

        np.testing.assert_array_equal(first, second)

    def test_invalid_structure_fails_before_loading(
        self, tmp_path: Path
    ) -> None:
        train = _make_split(tmp_path / "train", CLASSES)
        test = _make_split(tmp_path / "test", CLASSES[:2])

        with pytest.raises(ValueError):
            data_loader.load_datasets(train, test)