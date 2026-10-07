"""
This file downloads a subset of the Fruits-360 dataset and converts it into pixel art
(sharpen -> k-means palette quantization -> grid encoding).

Fruits-360 (https://github.com/fruits-360/fruits-360-100x100) by Mihai Oltean is
licensed under CC BY-SA 4.0. It is downloaded at runtime and not redistributed
with this example.
"""

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

FRUITS_360_URL = "https://github.com/fruits-360/fruits-360-100x100.git"
# Pin the dataset version so the example is reproducible
FRUITS_360_COMMIT = "c3b8394bc9cc903a82feb46c11aa1ad94b88a432"

# Fruits with distinct colors and shapes, so the generated classes are easy to tell apart
DEFAULT_CLASSES = [
    "Apple Granny Smith 1",
    "Banana 1",
    "Blueberry 1",
    "Orange 1",
    "Strawberry 1",
    "Eggplant 1",
]


def download_fruits_360(
    classes: list[str] = DEFAULT_CLASSES, data_dir: str | Path = "dataset/fruits-360"
) -> Path:
    """
    Download only the training and test images of the given classes using a sparse, blobless git checkout.
    Calling it again with other classes only fetches the missing ones.
    """
    data_dir = Path(data_dir)

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(data_dir), *args], check=True)

    if not (data_dir / ".git").exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        git("init", "--quiet")
        git("remote", "add", "origin", FRUITS_360_URL)
        git(
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "--filter=blob:none",
            "origin",
            FRUITS_360_COMMIT,
        )

    patterns = ["/LICENSE", "/README.md"]
    patterns += [f"/{split}/{c}/" for split in ["Training", "Test"] for c in classes]
    git("sparse-checkout", "set", "--no-cone", *patterns)
    git("checkout", "--quiet", FRUITS_360_COMMIT)
    return data_dir


def sharpen(img: Image.Image) -> Image.Image:
    """Sharpen edges with an unsharp mask so details survive the downscaling."""
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))


def fit_kmeans_palette(
    images: np.ndarray,
    n_colors: int,
    n_iter: int = 20,
    n_samples: int = 100_000,
    seed: int = 0,
) -> np.ndarray:
    """Fit one color palette shared by all images with k-means on a random sample of pixels."""
    rng = np.random.default_rng(seed)
    pixels = images.reshape(-1, 3).astype(np.float32)
    pixels = pixels[
        rng.choice(len(pixels), size=min(n_samples, len(pixels)), replace=False)
    ]

    # k-means++ initialization: spread the initial centers over the color space
    centers = [pixels[rng.integers(len(pixels))]]
    for _ in range(n_colors - 1):
        dist = np.min(
            np.sum((pixels[:, None] - np.array(centers)) ** 2, axis=-1), axis=1
        )
        centers.append(pixels[rng.choice(len(pixels), p=dist / dist.sum())])
    centers = np.array(centers)

    for _ in range(n_iter):
        assignment = palette_indices(pixels, centers)
        for k in range(n_colors):
            members = pixels[assignment == k]
            if len(members) > 0:
                centers[k] = members.mean(axis=0)
    return np.round(centers).astype(np.uint8)


def palette_indices(images: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Index of the nearest palette color for every pixel. Works on uint8 or float images in [0, 255]."""
    dist = np.sum(
        (images[..., None, :].astype(np.float32) - palette.astype(np.float32)) ** 2,
        axis=-1,
    )
    return np.argmin(dist, axis=-1)


def quantize_to_palette(images: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Replace every pixel by its nearest palette color. Works on uint8 or float images in [0, 255]."""
    return palette[palette_indices(images, palette)]


def grid_encode(indices: np.ndarray, size: int, n_colors: int) -> np.ndarray:
    """
    Encode a quantized image of shape (size * cell, size * cell) as a (size, size) grid
    of palette indices: each grid cell takes the most frequent palette color inside it.
    """
    cell = indices.shape[0] // size
    cells = (
        indices.reshape(size, cell, size, cell)
        .transpose(0, 2, 1, 3)
        .reshape(size, size, -1)
    )
    counts = np.eye(n_colors, dtype=np.int32)[cells].sum(axis=2)
    return np.argmax(counts, axis=-1)


def load_pixel_art_dataset(
    data_dir: str | Path,
    classes: list[str] = DEFAULT_CLASSES,
    split: str = "Training",
    size: int = 16,
    n_colors: int = 16,
    photo_size: int = 32,
    images_per_class: int | None = None,
    cell_size: int = 6,
    palette: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pairs of photos and their pixel art. The pixel art is made in three steps:
    1. Sharpen the photo, resized to (size * cell_size) pixels per side.
    2. Quantize it with a k-means color palette shared by all images.
    3. Grid encode it: every cell of cell_size x cell_size pixels becomes one pixel
       with the most frequent palette color of that cell.

    Pass the palette of the training split when loading the test split, so both use the same colors.

    Returns:
        pixel_art: uint8 array of shape (N, size, size, 3)
        photos: uint8 array of shape (N, photo_size, photo_size, 3), the downscaled photos
        labels: int64 array of shape (N,) indexing into `classes`
        palette: uint8 array of shape (n_colors, 3)
    """
    rng = np.random.default_rng(seed)
    resolution = size * cell_size
    sharpened, photos, labels = [], [], []
    for label, name in enumerate(classes):
        files = sorted((Path(data_dir) / split / name).glob("*.jpg"))
        if not files:
            raise FileNotFoundError(f"No images found for class '{name}' in {data_dir}")
        if images_per_class is not None:
            files = rng.choice(
                files, size=min(images_per_class, len(files)), replace=False
            )
        for f in files:
            img = Image.open(f).convert("RGB")
            photos.append(
                np.asarray(img.resize((photo_size, photo_size), Image.Resampling.BOX))
            )
            img = img.resize((resolution, resolution), Image.Resampling.BOX)
            sharpened.append(np.asarray(sharpen(img)))
            labels.append(label)

    sharpened = np.stack(sharpened)
    if palette is None:
        palette = fit_kmeans_palette(sharpened, n_colors, seed=seed)
    grids = np.stack(
        [
            grid_encode(palette_indices(p, palette), size, len(palette))
            for p in sharpened
        ]
    )
    return palette[grids], np.stack(photos), np.array(labels, dtype=np.int64), palette
