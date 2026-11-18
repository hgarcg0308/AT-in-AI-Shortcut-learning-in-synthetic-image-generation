"""
comun.py
========
Cosas que comparten todos los scripts: rutas, carga de datos, semillas y la
función que mide cuánto se parecen dos conjuntos de imágenes (correlación de
píxeles, igual que en el EDA).

Estructura esperada del proyecto (todo cuelga de la carpeta donde está este archivo):

    data/pneumoniamnist.npz          <- dataset original
    data/limpio/                     <- lo crea Limpieza.py
    outputs/                         <- lo crean DDPM.py y Verificador.py
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
RAIZ = Path(__file__).resolve().parent
RUTA_NPZ = RAIZ / "data" / "pneumoniamnist.npz"
DIR_LIMPIO = RAIZ / "data" / "limpio"
RUTA_LIMPIO = DIR_LIMPIO / "pneumoniamnist_limpio.npz"
RUTA_SUBMUESTRAS = DIR_LIMPIO / "submuestras.npz"
DIR_SALIDAS = RAIZ / "outputs"

NOMBRES_CLASE = ["normal", "neumonia"]  # etiqueta 0 y 1


# --------------------------------------------------------------------------
# Utilidades generales
# --------------------------------------------------------------------------
def fijar_semilla(semilla: int) -> None:
    """Deja todo reproducible (python, numpy y torch)."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


def elegir_dispositivo() -> torch.device:
    """GPU de NVIDIA si hay, si no GPU de Mac, si no CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Carga de datos
# --------------------------------------------------------------------------
def cargar_npz(ruta: Path) -> dict:
    """
    Lee un .npz con el formato de MedMNIST y devuelve un diccionario con
    train_x, train_y, val_x, val_y, test_x, test_y.
    Las imágenes quedan como uint8 (N, H, W) y las etiquetas como (N,).
    """
    f = np.load(ruta)
    datos = {}
    for split in ("train", "val", "test"):
        x = f[f"{split}_images"]
        if x.ndim == 4:  # (N, H, W, 1) -> (N, H, W)
            x = x[..., 0]
        datos[f"{split}_x"] = x.astype(np.uint8)
        datos[f"{split}_y"] = f[f"{split}_labels"].reshape(-1).astype(np.int64)
    return datos


def cargar_limpio() -> dict:
    """Carga el dataset ya limpio que genera Limpieza.py."""
    if not RUTA_LIMPIO.exists():
        raise FileNotFoundError(
            f"No encuentro {RUTA_LIMPIO}. Ejecuta primero:  python Limpieza.py"
        )
    return cargar_npz_limpio(RUTA_LIMPIO)


def cargar_npz_limpio(ruta: Path) -> dict:
    f = np.load(ruta)
    return {k: f[k] for k in f.files}


def cargar_submuestra(semilla: int):
    """
    Devuelve la submuestra del 10 % del train limpio para esa semilla.
    Retorna: x_10, y_10, x_resto, y_resto  (el resto es el 90 % que NO se toca
    salvo para el "techo").
    """
    datos = cargar_limpio()
    if not RUTA_SUBMUESTRAS.exists():
        raise FileNotFoundError(f"No encuentro {RUTA_SUBMUESTRAS}. Ejecuta Limpieza.py")
    sub = np.load(RUTA_SUBMUESTRAS)
    clave = f"idx_10pct_semilla{semilla}"
    if clave not in sub.files:
        disponibles = [k.replace("idx_10pct_semilla", "") for k in sub.files if k.startswith("idx_10pct")]
        raise KeyError(f"No existe la submuestra con semilla {semilla}. Disponibles: {disponibles}")
    idx = sub[clave]
    idx_resto = sub[f"idx_resto_semilla{semilla}"]
    x, y = datos["train_x"], datos["train_y"]
    return x[idx], y[idx], x[idx_resto], y[idx_resto]


# --------------------------------------------------------------------------
# Parecido entre imágenes (correlación de píxeles, como en el EDA)
# --------------------------------------------------------------------------
def estandarizar(X: np.ndarray) -> np.ndarray:
    """
    Aplana las imágenes y las normaliza para que el producto de dos filas sea
    directamente su correlación de Pearson (un valor entre -1 y 1).
    """
    Z = X.reshape(len(X), -1).astype(np.float32)
    Z -= Z.mean(axis=1, keepdims=True)
    Z /= Z.std(axis=1, keepdims=True) * np.sqrt(Z.shape[1]) + 1e-8
    return Z


def correlacion_maxima(A: np.ndarray, B: np.ndarray, bloque: int = 512):
    """
    Para cada imagen de A busca la imagen de B más parecida.
    Devuelve (r_max, indice_en_B), con r = correlación de píxeles.
    """
    ZA, ZB = estandarizar(A), estandarizar(B)
    r_max = np.empty(len(ZA), dtype=np.float32)
    vecino = np.empty(len(ZA), dtype=np.int64)
    for ini in range(0, len(ZA), bloque):
        R = ZA[ini:ini + bloque] @ ZB.T
        vecino[ini:ini + bloque] = R.argmax(axis=1)
        r_max[ini:ini + bloque] = R.max(axis=1)
    return r_max, vecino
