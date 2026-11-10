#!/usr/bin/env python3
"""
Diagnostico del entorno en dos partes, cada prueba en un proceso aparte (si una DLL falla y mata el
proceso, el diagnostico sigue y te dice cual fue):
  1. Imports de cada libreria y varios ordenes de import.
  2. Ejecucion de operaciones reales: torchvision, algebra lineal, PCA + regresion logistica,
     tests estadisticos, dibujar/guardar figuras y cargar PneumoniaMNIST (descarga si hace falta).

Uso (con el entorno AdvancedAITopics activo):  python check_env.py
"""
import subprocess
import sys

IMPORTS = {
    "numpy": "import numpy",
    "pandas": "import pandas",
    "scipy": "import scipy.stats",
    "scikit-learn": "import sklearn.linear_model, sklearn.decomposition",
    "matplotlib": "import matplotlib.pyplot",
    "seaborn": "import seaborn",
    "torch": "import torch",
    "torchvision": "import torchvision; from torchvision.datasets.utils import download_url",
    "medmnist": "import medmnist",
    "orden del script (medmnist antes que scipy/sklearn)":
        "import numpy, pandas, seaborn; from medmnist import PneumoniaMNIST; from scipy import stats; "
        "from sklearn.decomposition import PCA",
}

EJECUCION = {
    "numpy: algebra lineal (SVD y producto de matrices)":
        "import numpy as np; a = np.random.rand(300, 300); np.linalg.svd(a); a @ a",
    "scikit-learn: PCA + regresion logistica":
        "import numpy as np; from sklearn.decomposition import PCA; from sklearn.linear_model import LogisticRegression; "
        "X = np.random.rand(500, 100); y = (X[:, 0] > .5).astype(int); PCA(10).fit_transform(X); "
        "LogisticRegression(max_iter=500).fit(X, y)",
    "scipy: tests estadisticos (KS y chi-cuadrado)":
        "import numpy as np; from scipy import stats; stats.ks_2samp(np.random.rand(100), np.random.rand(100)); "
        "stats.chi2_contingency([[10, 20], [15, 25]])",
    "matplotlib + seaborn: dibujar y guardar una figura":
        "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, seaborn as sns, pandas as pd, "
        "numpy as np, tempfile, os; df = pd.DataFrame(dict(x=np.random.rand(50), g=['a', 'b'] * 25)); "
        "sns.violinplot(data=df, x='g', y='x'); plt.savefig(os.path.join(tempfile.gettempdir(), 'check_env.png'))",
    "medmnist: cargar PneumoniaMNIST (descarga si hace falta)":
        "import os; from medmnist import PneumoniaMNIST; os.makedirs('data', exist_ok=True); "
        "d = PneumoniaMNIST(split='test', download=True, root='data', size=28); print('imagenes en test:', len(d))",
}


def run(title, tests, timeout=300):
    print(f"\n== {title} ==")
    for name, code in tests.items():
        try:
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"{name:<58} SIN RESPUESTA tras {timeout} s")
            continue
        if r.returncode == 0:
            extra = f"  ({r.stdout.strip().splitlines()[-1]})" if r.stdout.strip() else ""
            print(f"{name:<58} OK{extra}")
        else:
            # En Windows un fallo de DLL aparece como codigo negativo; se muestra en hexadecimal.
            print(f"{name:<58} FALLA (codigo {r.returncode & 0xFFFFFFFF:#010x})")
            if r.stderr.strip():
                print("    " + r.stderr.strip().splitlines()[-1])


print(f"Python: {sys.executable} ({sys.version.split()[0]})")
print("Comprueba que la ruta pertenece al entorno AdvancedAITopics.")
run("1. Imports", IMPORTS)
run("2. Ejecucion de operaciones reales", EJECUCION)
