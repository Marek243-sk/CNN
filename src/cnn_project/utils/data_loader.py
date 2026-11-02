"""
This module is responsible for loading the static datasets (images)
and catch any mistakes related to data loading early on.
"""

from pathlib import Path

import tensorflow as tf

from cnn_project.constants import (
    BATCH_SIZE,
    IMAGE_SIZE,
    SEED,
    TEST_DATA_C,
    TRAIN_DATA_C,
)

_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".png" })

def get_class_names(directory: Path) -> list[str]:
    """
    Return the sorted class names (subdirectories names) found in `directory`.

    Args:
        direcotry (Path): Direcotry whose subdirectories are the classes.
    
    Returns:
        list[str]: Alphabetically sorted class names.
    
    Raises:
        FileNotFoundError: When the ``directory`` does not exist.
        ValueError: If direcotry does not contain any subfolders.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Direcotory {directory} does not exist.")

    names = sorted(
        path.name for path in directory.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
    )

    if not names:
        raise ValueError(f"No subfolders found in directory {directory}.")

    return names

def _count_images(directory: Path) -> int:
    """
    Count image files that sit directly inside `directory`.

    Args:
        directory (Path): Count images under this direcory.
    
    Returns:
        int: Number of image files.
    """
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )

def validate_dataset_structure(
    train_dir: Path,
    test_dir: Path,
) -> list[str]:
    """
    Validate that train and test contain the same non-empty class folders.
 
    Args:
        train_dir (Path): Directory with training class subfolders.
        test_dir (Path): Directory with test class subfolders.
 
    Returns:
        list[str]: Sorted class names shared by both splits.
 
    Raises:
        FileNotFoundError: When a split directory does not exist.
        ValueError: When the class folders differ between splits, or when
            any class folder contains no images.
    """
    train_classes = get_class_names(directory=train_dir)
    test_classes = get_class_names(directory=test_dir)

    if train_classes != test_classes:
        raise ValueError(
            "Train and Test structure does not match. "
            f"Train: {train_classes}\n"
            f"Test: {test_classes}"
        )

    empty = [
        str(split_dir / name)
        for split_dir in (train_dir, test_dir)
        for name in train_classes
        if _count_images(split_dir / name) == 0
    ]

    if empty:
        raise ValueError(f"Class folders contain no images: {empty}")

    return train_classes

def _load_split(
    directory: Path,
    class_names: list[str],
    *,
    shuffle: bool,
    validation_split: float | None = None,
    subset: str | None = None,
) -> tf.data.Dataset:
    """
    Load one dataset split from ``directory``.
 
    Args:
        directory (Path): Directory from which to load the split.
        class_names (list[str]): Explicit class order, so integer labels are
            guaranteed to be identical across splits.
        shuffle (bool): Whether to shuffle images.
        validation_split (float | None = None): Whether to use validation set and how big.
        subset (str | None):
 
    Returns:
        tf.data.Dataset: Batches of ``(images, int_labels)``.
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        class_names=class_names,
        label_mode="int",
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
        validation_split=validation_split,
        subset=subset,
    )
 
 
def load_datasets(
    train_dir: Path = TRAIN_DATA_C,
    test_dir: Path = TEST_DATA_C,
    validation_split: float = 0.2,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """
    Validate and load the ``Train`` and ``Test`` datasets.
 
    Args:
        train_dir (Path): Directory with training class subfolders.
        test_dir (Path): Directory with test class subfolders.
        validation_split (float): Fraction of the training set to use for validation set.
            Default is 20%.
 
    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
            Train dataset, test dataset and the class names, where the
            index of a name equals its integer label.
    """
    class_names = validate_dataset_structure(train_dir, test_dir)
 
    train_ds = _load_split(
        directory=train_dir,
        class_names=class_names,
        shuffle=True,
        validation_split=validation_split,
        subset="training",
    )

    val_ds = _load_split(
        directory=train_dir,
        class_names=class_names,
        shuffle=True,
        validation_split=validation_split,
        subset="validation",
    )

    test_ds = _load_split(
        directory=test_dir,
        class_names=class_names,
        shuffle=False
    )

    autotune = tf.data.AUTOTUNE
 
    return (
        train_ds.prefetch(autotune),
        val_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
        class_names,
    )
