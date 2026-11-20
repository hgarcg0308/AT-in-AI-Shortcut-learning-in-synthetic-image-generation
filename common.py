"""
common.py
=========
Cosas que comparten todos los scripts: rutas, carga de datos, semillas y la
función que mide cuánto se parecen dos conjuntos de imágenes (correlación de
píxeles, igual que en el EDA).

Estructura esperada del proyecto (todo cuelga de la carpeta donde está este archivo):

    data/pneumoniamnist.npz          <- dataset original
    data/clean/                      <- lo crea Cleaning.py
    outputs/                         <- lo crean DDPM.py y Verifier.py
"""

import os
# En Windows + conda, NumPy y PyTorch a veces cargan cada uno su copia de OpenMP y
# Python se cierra con "OMP: Error #15". Esta línea lo evita. Debe ir ANTES de importar nada más.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
NPZ_PATH = ROOT / "data" / "pneumoniamnist.npz"
CLEAN_DIR = ROOT / "data" / "clean"
CLEAN_PATH = CLEAN_DIR / "pneumoniamnist_clean.npz"
SUBSAMPLES_PATH = CLEAN_DIR / "subsamples.npz"
OUTPUT_DIR = ROOT / "outputs"

CLASS_NAMES = ["normal", "pneumonia"]  # etiqueta 0 y 1


# --------------------------------------------------------------------------
# Utilidades generales
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Deja todo reproducible (python, numpy y torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    """GPU de NVIDIA si hay, si no GPU de Mac, si no CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Carga de datos
# --------------------------------------------------------------------------
def load_npz(path: Path) -> dict:
    """
    Lee un .npz con el formato de MedMNIST y devuelve un diccionario con
    train_x, train_y, val_x, val_y, test_x, test_y.
    Las imágenes quedan como uint8 (N, H, W) y las etiquetas como (N,).
    """
    f = np.load(path)
    data = {}
    for split in ("train", "val", "test"):
        x = f[f"{split}_images"]
        if x.ndim == 4:  # (N, H, W, 1) -> (N, H, W)
            x = x[..., 0]
        data[f"{split}_x"] = x.astype(np.uint8)
        data[f"{split}_y"] = f[f"{split}_labels"].reshape(-1).astype(np.int64)
    return data


def load_clean() -> dict:
    """Carga el dataset ya limpio que genera Cleaning.py."""
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {CLEAN_PATH}. Run first:  python Cleaning.py"
        )
    return load_clean_npz(CLEAN_PATH)


def load_clean_npz(path: Path) -> dict:
    f = np.load(path)
    return {k: f[k] for k in f.files}


def load_subsample(seed: int):
    """
    Devuelve la submuestra del 10 % del train limpio para esa semilla.
    Retorna: x_10, y_10, x_rest, y_rest  (el resto es el 90 % que NO se toca
    salvo para el "techo").
    """
    data = load_clean()
    if not SUBSAMPLES_PATH.exists():
        raise FileNotFoundError(f"Cannot find {SUBSAMPLES_PATH}. Run Cleaning.py")
    sub = np.load(SUBSAMPLES_PATH)
    key = f"idx_10pct_seed{seed}"
    if key not in sub.files:
        available = [k.replace("idx_10pct_seed", "") for k in sub.files if k.startswith("idx_10pct")]
        raise KeyError(f"No subsample exists for seed {seed}. Available: {available}")
    idx = sub[key]
    idx_rest = sub[f"idx_rest_seed{seed}"]
    x, y = data["train_x"], data["train_y"]
    return x[idx], y[idx], x[idx_rest], y[idx_rest]


# --------------------------------------------------------------------------
# Parecido entre imágenes (correlación de píxeles, como en el EDA)
# --------------------------------------------------------------------------
def standardize(X: np.ndarray) -> np.ndarray:
    """
    Aplana las imágenes y las normaliza para que el producto de dos filas sea
    directamente su correlación de Pearson (un valor entre -1 y 1).
    """
    Z = X.reshape(len(X), -1).astype(np.float32)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= Z.std(axis=1, keepdims=True) * np.sqrt(Z.shape[1]) + 1e-8
    return Z


def max_correlation(A: np.ndarray, B: np.ndarray, block: int = 512):
    """
    Para cada imagen de A busca la imagen de B más parecida.
    Devuelve (r_max, indice_en_B), con r = correlación de píxeles.
    """
    ZA, ZB = standardize(A), standardize(B)
    r_max = np.empty(len(ZA), dtype=np.float32)
    neighbor = np.empty(len(ZA), dtype=np.int64)
    for start in range(0, len(ZA), block):
        R = ZA[start:start + block] @ ZB.T
        neighbor[start:start + block] = R.argmax(axis=1)
        r_max[start:start + block] = R.max(axis=1)
    return r_max, neighbor
