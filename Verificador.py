"""
Verificador.py
==============
Fases 3 (clasificador base) y 5 (control de calidad de lo generado) del flujo.

El "verificador" es una CNN pequeña que se entrena SOLO con el 10 % real y que
luego sirve para revisar las imágenes que genera el DDPM. Se usa como
verificador, no como filtro: no decide qué entra en el entrenamiento, solo nos
dice si las imágenes sintéticas son de fiar.

Qué revisa de las imágenes sintéticas:
  1. Ruido de etiquetas: ¿el verificador está de acuerdo con la clase que se pidió?
     (se compara con lo que falla en imágenes reales de validación, para tener referencia)
  2. Memorización: ¿las sintéticas son copias del train? Se compara su parecido
     con el train frente al parecido que tienen las imágenes reales de val.
  3. Calidad global (KID): ¿se parecen en conjunto a las reales? Se calcula con las
     características internas del verificador.
  4. Guarda por imagen su probabilidad, entropía y margen, que servirán luego para
     la selección por incertidumbre (Fase 6).

Uso:
    # 1) Entrenar el verificador con la submuestra del 10 % de la semilla 0
    python Verificador.py entrenar --semilla 0

    # 2) Evaluar un archivo de sintéticas generado con DDPM.py
    python Verificador.py evaluar --semilla 0 --sinteticas outputs/ddpm/sinteticas_semilla0_w2.0.npz

Salidas en outputs/verificador/.
"""

import os
# En Windows + conda, NumPy y PyTorch a veces cargan cada uno su copia de OpenMP y
# Python se cierra con "OMP: Error #15". Esta línea lo evita. Debe ir ANTES de importar nada más.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from comun import (DIR_SALIDAS, cargar_limpio, cargar_submuestra, correlacion_maxima,
                   elegir_dispositivo, fijar_semilla)

DIR_VERIF = DIR_SALIDAS / "verificador"


# ==========================================================================
# 1. La red
# ==========================================================================
def bloque(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNVerificadora(nn.Module):
    """Tres bloques de convolución + una capa densa. Saca un único número (logit) por imagen."""

    def __init__(self, canales=32, dropout=0.3):
        super().__init__()
        self.cuerpo = nn.Sequential(bloque(1, canales), bloque(canales, canales * 2), bloque(canales * 2, canales * 4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(canales * 4, 64)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def caracteristicas(self, x):
        """Vector interno de 64 números que resume la imagen (lo usamos para el KID)."""
        h = self.pool(self.cuerpo(x)).flatten(1)
        return F.relu(self.fc1(h))

    def forward(self, x):
        return self.fc2(self.drop(self.caracteristicas(x))).squeeze(1)


def a_tensor(x_uint8: np.ndarray, dev) -> torch.Tensor:
    """(N,H,W) uint8 -> (N,1,H,W) float en [0,1]."""
    return torch.from_numpy(x_uint8).float().div(255).unsqueeze(1).to(dev)


def desplazar(x: torch.Tensor, max_px: int = 2) -> torch.Tensor:
    """Aumentación suave: mueve cada imagen unos pocos píxeles al azar."""
    b, _, h, w = x.shape
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = (torch.rand(b, device=x.device) * 2 - 1) * max_px * 2 / w
    theta[:, 1, 2] = (torch.rand(b, device=x.device) * 2 - 1) * max_px * 2 / h
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


@torch.no_grad()
def predecir(modelo, x_uint8, dev, batch=512):
    """Devuelve probabilidad de neumonía y características internas de cada imagen."""
    modelo.eval()
    probs, feats = [], []
    for i in range(0, len(x_uint8), batch):
        x = a_tensor(x_uint8[i:i + batch], dev)
        f = modelo.caracteristicas(x)
        probs.append(torch.sigmoid(modelo.fc2(f).squeeze(1)).cpu())
        feats.append(f.cpu())
    return torch.cat(probs).numpy(), torch.cat(feats).numpy()


def metricas(y, p, umbral=0.5) -> dict:
    """AUC y, por clase, cuántas acierta (sensibilidad = neumonía, especificidad = normal)."""
    pred = (p >= umbral).astype(int)
    out = {"auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
           "accuracy": float((pred == y).mean())}
    out["sensibilidad"] = float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
    out["especificidad"] = float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
    return out


# ==========================================================================
# 2. Entrenamiento del verificador
# ==========================================================================
def entrenar(args):
    dev = elegir_dispositivo()
    fijar_semilla(args.semilla)
    DIR_VERIF.mkdir(parents=True, exist_ok=True)

    x10, y10, _, _ = cargar_submuestra(args.semilla)          # 10 % real
    datos = cargar_limpio()
    xva, yva = datos["val_x"], datos["val_y"]                  # validación limpia
    xte, yte = datos["test_x"], datos["test_y"]                # test oficial (solo para informar)
    print(f"Dispositivo: {dev} | train 10 %: {len(x10)} | val limpia: {len(xva)} | test: {len(xte)}")

    X, Y = a_tensor(x10, dev), torch.from_numpy(y10).float().to(dev)
    modelo = CNNVerificadora(args.canales, args.dropout).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-3)
    # como hay ~3 veces más neumonías que normales, damos más peso a la clase minoritaria
    pos_weight = torch.tensor((y10 == 0).sum() / max(1, (y10 == 1).sum()), device=dev).float()
    criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    mejor_auc, mejor_loss, mejor_estado, sin_mejora = -1.0, float('inf'), None, 0
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        orden = torch.randperm(len(X), device=dev)
        perdida = 0.0
        for i in range(0, len(X), args.batch):
            idx = orden[i:i + args.batch]
            if len(idx) < 2:  # BatchNorm no funciona con 1 sola imagen
                continue
            xb = X[idx] if args.sin_aug else desplazar(X[idx])
            loss = criterio(modelo(xb), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            perdida += loss.item() * len(idx)

        # early stopping: nos quedamos con la época de mejor AUC en la validación limpia
        p_va, _ = predecir(modelo, xva, dev)
        auc_va = roc_auc_score(yva, p_va)
        pc = np.clip(p_va, 1e-6, 1 - 1e-6)
        loss_va = float(-(yva * np.log(pc) + (1 - yva) * np.log(1 - pc)).mean())
        # mejor = más AUC; si el AUC empata (pasa cuando ya es ~1.0), gana la menor pérdida
        if auc_va > mejor_auc + 1e-4 or (abs(auc_va - mejor_auc) <= 1e-4 and loss_va < mejor_loss):
            mejor_auc, mejor_loss, sin_mejora = max(auc_va, mejor_auc), loss_va, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
        if epoca % 5 == 0 or epoca == 1:
            print(f"época {epoca:3d} | pérdida {perdida / len(X):.4f} | AUC val {auc_va:.3f} (mejor {mejor_auc:.3f})")
        if sin_mejora >= args.paciencia:
            print(f"Parada anticipada en la época {epoca}")
            break

    modelo.load_state_dict(mejor_estado)
    p_va, _ = predecir(modelo, xva, dev)
    p_te, _ = predecir(modelo, xte, dev)
    res = {"semilla": args.semilla, "val": metricas(yva, p_va), "test": metricas(yte, p_te)}
    print("\nResultados (umbral 0.5):")
    for nombre in ("val", "test"):
        m = res[nombre]
        print(f"  {nombre:4s} AUC {m['auc']:.3f} | sensibilidad {m['sensibilidad']:.3f} | "
              f"especificidad {m['especificidad']:.3f} | accuracy {m['accuracy']:.3f}")

    torch.save({"estado": modelo.state_dict(), "canales": args.canales, "dropout": args.dropout,
                "semilla": args.semilla}, DIR_VERIF / f"verificador_semilla{args.semilla}.pt")
    with open(DIR_VERIF / f"verificador_semilla{args.semilla}_metricas.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"Guardado en {DIR_VERIF}")


def cargar_verificador(semilla, dev):
    ruta = DIR_VERIF / f"verificador_semilla{semilla}.pt"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Entrena antes:  python Verificador.py entrenar --semilla {semilla}")
    ck = torch.load(ruta, map_location=dev)
    m = CNNVerificadora(ck["canales"], ck["dropout"]).to(dev)
    m.load_state_dict(ck["estado"])
    return m.eval()


# ==========================================================================
# 3. KID (parecido entre dos conjuntos de imágenes)
# ==========================================================================
def kid(fa: np.ndarray, fb: np.ndarray, n_subconjuntos=50, tam=100, semilla=0):
    """
    Kernel Inception Distance, pero con las características del verificador en vez
    de Inception (nuestras imágenes son pequeñas y en gris). Cuanto más cerca de 0,
    más se parecen los dos conjuntos. Funciona bien con pocas imágenes (a diferencia
    del FID). Devuelve (media, desviación) sobre varios subconjuntos al azar.
    """
    rng = np.random.default_rng(semilla)
    d = fa.shape[1]
    m = min(len(fa), len(fb), tam)
    if m < 5:
        return float("nan"), float("nan")

    def k(a, b):  # kernel polinómico
        return (a @ b.T / d + 1.0) ** 3

    valores = []
    for _ in range(n_subconjuntos):
        a = fa[rng.choice(len(fa), m, replace=False)]
        b = fb[rng.choice(len(fb), m, replace=False)]
        kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
        mmd = ((kaa.sum() - np.trace(kaa)) / (m * (m - 1)) +
               (kbb.sum() - np.trace(kbb)) / (m * (m - 1)) - 2 * kab.mean())
        valores.append(mmd)
    return float(np.mean(valores)), float(np.std(valores))


# ==========================================================================
# 4. Evaluación de las imágenes sintéticas
# ==========================================================================
def evaluar(args):
    dev = elegir_dispositivo()
    DIR_VERIF.mkdir(parents=True, exist_ok=True)
    ruta_sint = Path(args.sinteticas)
    s = np.load(ruta_sint)
    xs, ys = s["images"], s["labels"].astype(int)
    nombre = ruta_sint.stem

    modelo = cargar_verificador(args.semilla, dev)
    x10, y10, _, _ = cargar_submuestra(args.semilla)
    datos = cargar_limpio()
    xva, yva = datos["val_x"], datos["val_y"]
    print(f"Sintéticas: {len(xs)} (normal {(ys == 0).sum()}, neumonía {(ys == 1).sum()}) | referencia real: {len(x10)} imágenes del 10 %")

    p_s, f_s = predecir(modelo, xs, dev)
    p_va, f_va = predecir(modelo, xva, dev)
    _, f_10 = predecir(modelo, x10, dev)
    res = {"archivo": ruta_sint.name, "n_sinteticas": int(len(xs))}

    # ---------- 1) Ruido de etiquetas ----------
    pred_s = (p_s >= 0.5).astype(int)
    desacuerdo = {"normal": float((pred_s[ys == 0] != 0).mean()) if (ys == 0).any() else float("nan"),
                  "neumonia": float((pred_s[ys == 1] != 1).mean()) if (ys == 1).any() else float("nan")}
    pred_va = (p_va >= 0.5).astype(int)
    error_real = {"normal": float((pred_va[yva == 0] != 0).mean()), "neumonia": float((pred_va[yva == 1] != 1).mean())}
    res["ruido_etiquetas"] = {"desacuerdo_sinteticas": desacuerdo, "error_en_val_real_(referencia)": error_real}
    print("\n[1] Ruido de etiquetas (el verificador NO está de acuerdo con la clase pedida)")
    for c in ("normal", "neumonia"):
        print(f"    {c:9s}: sintéticas {desacuerdo[c] * 100:5.1f} %   | val real (referencia) {error_real[c] * 100:5.1f} %")

    # ---------- 2) Memorización ----------
    r_sint, vec_sint = correlacion_maxima(xs, x10)
    r_val, _ = correlacion_maxima(xva, x10)
    p05_val = float(np.percentile(r_val, 95))  # los "más parecidos" que se ven entre reales distintas
    res["memorizacion"] = {
        "r_max_mediana_sinteticas": float(np.median(r_sint)), "r_max_mediana_val_real": float(np.median(r_val)),
        "frac_sinteticas_r_mayor_0.99": float((r_sint > 0.99).mean()),
        "frac_sinteticas_por_encima_percentil95_val": float((r_sint > p05_val).mean()),
        "percentil95_val": p05_val,
    }
    print("\n[2] Memorización (correlación con su vecina más cercana del train 10 %)")
    print(f"    mediana  sintéticas {np.median(r_sint):.3f} | val real {np.median(r_val):.3f}")
    print(f"    sintéticas con r > 0.99 (casi copias): {(r_sint > 0.99).mean() * 100:.1f} %")
    print(f"    sintéticas más parecidas al train que el 95 % de val: {(r_sint > p05_val).mean() * 100:.1f} % (esperado ~5 %)")
    if np.median(r_sint) > np.median(r_val) + 0.05 or (r_sint > 0.99).mean() > 0.02:
        print("    AVISO: las sintéticas se parecen al train más de lo normal; el modelo podría estar copiando.")

    # ---------- 3) KID por clase ----------
    res["kid"] = {}
    print("\n[3] KID (más bajo = más parecido a las reales del 10 %)")
    for c, nom in ((0, "normal"), (1, "neumonia")):
        if (ys == c).sum() < 5:
            continue
        k_sint = kid(f_s[ys == c], f_10[y10 == c])
        k_ref = kid(f_va[yva == c], f_10[y10 == c])  # referencia: val real vs train real
        res["kid"][nom] = {"sinteticas_vs_train10": k_sint, "val_real_vs_train10_(referencia)": k_ref}
        print(f"    {nom:9s}: sintéticas {k_sint[0]:.4f} ± {k_sint[1]:.4f} | val real (referencia) {k_ref[0]:.4f} ± {k_ref[1]:.4f}")
    print("    (el verificador se entrenó con el train 10 %, así que el KID de referencia sale algo optimista)")

    # ---------- 4) Incertidumbre por imagen (para la Fase 6) ----------
    p_c = np.clip(p_s, 1e-6, 1 - 1e-6)
    entropia = -(p_c * np.log(p_c) + (1 - p_c) * np.log(1 - p_c))
    margen = np.abs(2 * p_s - 1)
    np.savez_compressed(DIR_VERIF / f"evaluacion_{nombre}.npz", p_neumonia=p_s, etiqueta_pedida=ys,
                        entropia=entropia, margen=margen, r_max_train=r_sint, vecino_train=vec_sint)
    with open(DIR_VERIF / f"evaluacion_{nombre}.json", "w") as f:
        json.dump(res, f, indent=2)

    # ---------- Figuras ----------
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(r_val, bins=30, alpha=0.6, density=True, label="val real → train 10 %")
    ax[0].hist(r_sint, bins=30, alpha=0.6, density=True, label="sintéticas → train 10 %")
    ax[0].set_xlabel("correlación con la imagen más parecida del train")
    ax[0].set_title("Memorización")
    ax[0].legend(fontsize=8)
    ax[1].hist(p_s[ys == 0], bins=20, alpha=0.6, label="pedidas normal")
    ax[1].hist(p_s[ys == 1], bins=20, alpha=0.6, label="pedidas neumonía")
    ax[1].set_xlabel("P(neumonía) según el verificador")
    ax[1].set_title("¿Está de acuerdo con la clase pedida?")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DIR_VERIF / f"evaluacion_{nombre}_histogramas.png", dpi=130)
    plt.close(fig)

    # las 6 sintéticas más parecidas al train, con su vecina
    top = np.argsort(-r_sint)[:6]
    fig, ax = plt.subplots(2, 6, figsize=(10, 3.6))
    for j, i in enumerate(top):
        ax[0, j].imshow(xs[i], cmap="gray", vmin=0, vmax=255)
        ax[0, j].set_title(f"r={r_sint[i]:.3f}", fontsize=8)
        ax[1, j].imshow(x10[vec_sint[i]], cmap="gray", vmin=0, vmax=255)
        ax[0, j].axis("off")
        ax[1, j].axis("off")
    ax[0, 0].text(-0.1, 0.5, "sintética", transform=ax[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    ax[1, 0].text(-0.1, 0.5, "más cercana\nen train", transform=ax[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(DIR_VERIF / f"evaluacion_{nombre}_mas_parecidas.png", dpi=130)
    plt.close(fig)
    print(f"\nResultados guardados en {DIR_VERIF} (evaluacion_{nombre}.*)")


def main():
    ap = argparse.ArgumentParser(description="CNN verificadora y control de calidad de imágenes sintéticas")
    sub = ap.add_subparsers(dest="comando", required=True)

    e = sub.add_parser("entrenar", help="entrena el verificador con el 10 %% real")
    e.add_argument("--semilla", type=int, default=0, help="qué submuestra del 10 %% usar")
    e.add_argument("--epocas", type=int, default=100)
    e.add_argument("--paciencia", type=int, default=15, help="épocas sin mejorar antes de parar")
    e.add_argument("--batch", type=int, default=32)
    e.add_argument("--lr", type=float, default=1e-3)
    e.add_argument("--canales", type=int, default=32)
    e.add_argument("--dropout", type=float, default=0.3)
    e.add_argument("--sin-aug", action="store_true", help="desactiva los pequeños desplazamientos")
    e.set_defaults(func=entrenar)

    v = sub.add_parser("evaluar", help="revisa un archivo de imágenes sintéticas")
    v.add_argument("--semilla", type=int, default=0, help="verificador y submuestra de referencia")
    v.add_argument("--sinteticas", required=True, help="ruta al .npz generado por DDPM.py")
    v.set_defaults(func=evaluar)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
