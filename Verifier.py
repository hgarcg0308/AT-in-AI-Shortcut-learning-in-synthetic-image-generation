"""
Verifier.py
===========
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
  4. Precisión y recall de distribuciones (en el espacio de características del
     verificador): precisión = ¿caen las sintéticas dentro de lo que cubre lo real?
     (fidelidad); recall = ¿cubren las sintéticas lo que cubre lo real? (diversidad).
     Se compara con la referencia val real vs train 10 %.
  5. Guarda por imagen su probabilidad, entropía y margen, y la incertidumbre
     epistémica por MC-dropout (información mutua), que sirven luego para la
     selección por incertidumbre (Fase 6, Scenarios.py).

Uso:
    # 1) Entrenar el verificador con la submuestra del 10 % de la semilla 0
    python Verifier.py train --seed 0

    # 2) Evaluar un archivo de sintéticas generado con DDPM.py
    python Verifier.py evaluate --seed 0 --synthetic outputs/ddpm/synthetic_seed0_w2.0.npz

Salidas en outputs/verifier/.
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

from common import (OUTPUT_DIR, load_clean, load_subsample, max_correlation,
                    pick_device, set_seed)

VERIFIER_DIR = OUTPUT_DIR / "verifier"


# ==========================================================================
# 1. La red
# ==========================================================================
def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class VerifierCNN(nn.Module):
    """Tres bloques de convolución + una capa densa. Saca un único número (logit) por imagen."""

    def __init__(self, channels=32, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(block(1, channels), block(channels, channels * 2), block(channels * 2, channels * 4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels * 4, 64)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def features(self, x):
        """Vector interno de 64 números que resume la imagen (lo usamos para el KID)."""
        h = self.pool(self.body(x)).flatten(1)
        return F.relu(self.fc1(h))

    def forward(self, x):
        return self.fc2(self.drop(self.features(x))).squeeze(1)


def to_tensor(x_uint8: np.ndarray, dev) -> torch.Tensor:
    """(N,H,W) uint8 -> (N,1,H,W) float en [0,1]."""
    return torch.from_numpy(x_uint8).float().div(255).unsqueeze(1).to(dev)


def shift(x: torch.Tensor, max_px: int = 2) -> torch.Tensor:
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
def predict(model, x_uint8, dev, batch=512):
    """Devuelve probabilidad de neumonía y características internas de cada imagen."""
    model.eval()
    probs, feats = [], []
    for i in range(0, len(x_uint8), batch):
        x = to_tensor(x_uint8[i:i + batch], dev)
        f = model.features(x)
        probs.append(torch.sigmoid(model.fc2(f).squeeze(1)).cpu())
        feats.append(f.cpu())
    return torch.cat(probs).numpy(), torch.cat(feats).numpy()


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Entropía (en nats) de una probabilidad de clase binaria."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


@torch.no_grad()
def predict_mc(model, x_uint8, dev, n_passes=30, batch=512) -> dict:
    """
    MC-dropout: repite la predicción n_passes veces con el dropout ACTIVO (BatchNorm en
    modo evaluación) y mide cuánto cambia. Devuelve, por imagen:
        p_mean          probabilidad media de neumonía
        std             desviación típica de la probabilidad entre pasadas
        entropy         entropía predictiva  H[ p_mean ]            (incertidumbre total)
        expected_entropy media de H[ p_t ] entre pasadas            (parte aleatoria)
        mutual_info     entropy - expected_entropy                  (parte epistémica, BALD)
    Limitación: el modelo solo tiene dropout en la cabeza densa (sobre el vector de 64
    características), así que la incertidumbre epistémica que se mide es la de esa cabeza,
    no la de toda la red. Sirve para ordenar imágenes, no como estimación calibrada.
    """
    model.eval()
    model.drop.train()  # solo el dropout activo
    chunks = []
    try:
        for i in range(0, len(x_uint8), batch):
            f = model.features(to_tensor(x_uint8[i:i + batch], dev))  # determinista (BatchNorm en eval)
            chunks.append(torch.stack([torch.sigmoid(model.fc2(model.drop(f)).squeeze(1))
                                       for _ in range(n_passes)]).cpu())
    finally:
        model.eval()
    P = torch.cat(chunks, dim=1).numpy()  # (n_passes, N)
    total, expected = binary_entropy(P.mean(axis=0)), binary_entropy(P).mean(axis=0)
    return {"p_mean": P.mean(axis=0), "std": P.std(axis=0), "entropy": total,
            "expected_entropy": expected, "mutual_info": np.maximum(total - expected, 0.0)}


def metrics(y, p, threshold=0.5) -> dict:
    """AUC y, por clase, cuántas acierta (sensibilidad = neumonía, especificidad = normal)."""
    pred = (p >= threshold).astype(int)
    out = {"auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
           "accuracy": float((pred == y).mean())}
    out["sensitivity"] = float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
    out["specificity"] = float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
    return out


# ==========================================================================
# 2. Entrenamiento del verificador
# ==========================================================================
def train(args):
    dev = pick_device()
    set_seed(args.seed)
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)

    x10, y10, _, _ = load_subsample(args.seed)                 # 10 % real
    data = load_clean()
    xva, yva = data["val_x"], data["val_y"]                    # validación limpia
    xte, yte = data["test_x"], data["test_y"]                  # test oficial (solo para informar)
    print(f"Device: {dev} | train 10%: {len(x10)} | clean val: {len(xva)} | test: {len(xte)}")

    X, Y = to_tensor(x10, dev), torch.from_numpy(y10).float().to(dev)
    model = VerifierCNN(args.channels, args.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    # como hay ~3 veces más neumonías que normales, damos más peso a la clase minoritaria
    pos_weight = torch.tensor((y10 == 0).sum() / max(1, (y10 == 1).sum()), device=dev).float()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_loss, best_state, no_improve = -1.0, float('inf'), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(X), device=dev)
        total_loss = 0.0
        for i in range(0, len(X), args.batch):
            idx = order[i:i + args.batch]
            if len(idx) < 2:  # BatchNorm no funciona con 1 sola imagen
                continue
            xb = X[idx] if args.no_aug else shift(X[idx])
            loss = criterion(model(xb), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        # early stopping: nos quedamos con la época de mejor AUC en la validación limpia
        p_va, _ = predict(model, xva, dev)
        auc_va = roc_auc_score(yva, p_va)
        pc = np.clip(p_va, 1e-6, 1 - 1e-6)
        loss_va = float(-(yva * np.log(pc) + (1 - yva) * np.log(1 - pc)).mean())
        # mejor = más AUC; si el AUC empata (pasa cuando ya es ~1.0), gana la menor pérdida
        if auc_va > best_auc + 1e-4 or (abs(auc_va - best_auc) <= 1e-4 and loss_va < best_loss):
            best_auc, best_loss, no_improve = max(auc_va, best_auc), loss_va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | loss {total_loss / len(X):.4f} | val AUC {auc_va:.3f} (best {best_auc:.3f})")
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    p_va, _ = predict(model, xva, dev)
    p_te, _ = predict(model, xte, dev)
    res = {"seed": args.seed, "val": metrics(yva, p_va), "test": metrics(yte, p_te)}
    print("\nResults (threshold 0.5):")
    for name in ("val", "test"):
        m = res[name]
        print(f"  {name:4s} AUC {m['auc']:.3f} | sensitivity {m['sensitivity']:.3f} | "
              f"specificity {m['specificity']:.3f} | accuracy {m['accuracy']:.3f}")

    torch.save({"state": model.state_dict(), "channels": args.channels, "dropout": args.dropout,
                "seed": args.seed}, VERIFIER_DIR / f"verifier_seed{args.seed}.pt")
    with open(VERIFIER_DIR / f"verifier_seed{args.seed}_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved in {VERIFIER_DIR}")


def load_verifier(seed, dev):
    path = VERIFIER_DIR / f"verifier_seed{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Train first:  python Verifier.py train --seed {seed}")
    ck = torch.load(path, map_location=dev)
    m = VerifierCNN(ck["channels"], ck["dropout"]).to(dev)
    m.load_state_dict(ck["state"])
    return m.eval()


# ==========================================================================
# 3. KID (parecido entre dos conjuntos de imágenes)
# ==========================================================================
def kid(fa: np.ndarray, fb: np.ndarray, n_subsets=50, size=100, seed=0):
    """
    Kernel Inception Distance, pero con las características del verificador en vez
    de Inception (nuestras imágenes son pequeñas y en gris). Cuanto más cerca de 0,
    más se parecen los dos conjuntos. Funciona bien con pocas imágenes (a diferencia
    del FID). Devuelve (media, desviación) sobre varios subconjuntos al azar.
    """
    rng = np.random.default_rng(seed)
    d = fa.shape[1]
    m = min(len(fa), len(fb), size)
    if m < 5:
        return float("nan"), float("nan")

    def k(a, b):  # kernel polinómico
        return (a @ b.T / d + 1.0) ** 3

    values = []
    for _ in range(n_subsets):
        a = fa[rng.choice(len(fa), m, replace=False)]
        b = fb[rng.choice(len(fb), m, replace=False)]
        kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
        mmd = ((kaa.sum() - np.trace(kaa)) / (m * (m - 1)) +
               (kbb.sum() - np.trace(kbb)) / (m * (m - 1)) - 2 * kab.mean())
        values.append(mmd)
    return float(np.mean(values)), float(np.std(values))


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distancias euclídeas entre todas las filas de a y de b."""
    d = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(d, 0.0))


def precision_recall(f_real: np.ndarray, f_fake: np.ndarray, k=3, n_repeats=10, seed=0) -> dict:
    """
    Precisión y recall de distribuciones (Kynkäänniemi et al., 2019) con k vecinos.
    El "soporte" de cada conjunto es la unión de bolas centradas en sus puntos, con radio la
    distancia a su k-ésimo vecino.
        precisión = fracción de puntos de f_fake que caen dentro del soporte de f_real
        recall    = fracción de puntos de f_real que caen dentro del soporte de f_fake
    Los dos conjuntos se submuestrean al mismo tamaño (el del más pequeño) y se promedia sobre
    n_repeats, porque el soporte de un conjunto grande es mayor solo por tener más puntos.
    Devuelve {"precision": [media, sd], "recall": [media, sd], "n_per_set": m}.
    """
    rng = np.random.default_rng(seed)
    m = min(len(f_real), len(f_fake))
    if m <= k + 1:
        nan = float("nan")
        return {"precision": [nan, nan], "recall": [nan, nan], "n_per_set": int(m)}
    precisions, recalls = [], []
    for _ in range(n_repeats):
        a = f_real[rng.choice(len(f_real), m, replace=False)].astype(np.float64)
        b = f_fake[rng.choice(len(f_fake), m, replace=False)].astype(np.float64)
        d_ab = _distances(a, b)
        r_a = np.sort(_distances(a, a), axis=1)[:, k]  # la columna 0 es el propio punto
        r_b = np.sort(_distances(b, b), axis=1)[:, k]
        precisions.append(float((d_ab <= r_a[:, None]).any(axis=0).mean()))
        recalls.append(float((d_ab <= r_b[None, :]).any(axis=1).mean()))
    return {"precision": [float(np.mean(precisions)), float(np.std(precisions))],
            "recall": [float(np.mean(recalls)), float(np.std(recalls))], "n_per_set": int(m)}


def uncertainty_summary(mc: dict, labels: np.ndarray) -> dict:
    """Medias de la incertidumbre MC-dropout, en total y por clase."""
    def one(mask):
        return {"mean_mutual_info": float(mc["mutual_info"][mask].mean()),
                "mean_predictive_entropy": float(mc["entropy"][mask].mean()),
                "mean_mc_std": float(mc["std"][mask].mean()), "n": int(mask.sum())}
    out = {"all": one(np.ones(len(labels), dtype=bool))}
    for c, name in ((0, "normal"), (1, "pneumonia")):
        if (labels == c).any():
            out[name] = one(labels == c)
    return out


# ==========================================================================
# 4. Evaluación de las imágenes sintéticas
# ==========================================================================
def evaluate(args):
    dev = pick_device()
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    synth_path = Path(args.synthetic)
    s = np.load(synth_path)
    xs, ys = s["images"], s["labels"].astype(int)
    name = synth_path.stem

    model = load_verifier(args.seed, dev)
    x10, y10, _, _ = load_subsample(args.seed)
    data = load_clean()
    xva, yva = data["val_x"], data["val_y"]
    print(f"Synthetic: {len(xs)} (normal {(ys == 0).sum()}, pneumonia {(ys == 1).sum()}) | real reference: {len(x10)} images of the 10%")

    p_s, f_s = predict(model, xs, dev)
    p_va, f_va = predict(model, xva, dev)
    _, f_10 = predict(model, x10, dev)
    res = {"file": synth_path.name, "n_synthetic": int(len(xs))}

    # ---------- 1) Ruido de etiquetas ----------
    pred_s = (p_s >= 0.5).astype(int)
    disagreement = {"normal": float((pred_s[ys == 0] != 0).mean()) if (ys == 0).any() else float("nan"),
                    "pneumonia": float((pred_s[ys == 1] != 1).mean()) if (ys == 1).any() else float("nan")}
    pred_va = (p_va >= 0.5).astype(int)
    real_error = {"normal": float((pred_va[yva == 0] != 0).mean()), "pneumonia": float((pred_va[yva == 1] != 1).mean())}
    res["label_noise"] = {"synthetic_disagreement": disagreement, "error_on_real_val_(reference)": real_error}
    print("\n[1] Label noise (the verifier does NOT agree with the requested class)")
    for c in ("normal", "pneumonia"):
        print(f"    {c:9s}: synthetic {disagreement[c] * 100:5.1f} %   | real val (reference) {real_error[c] * 100:5.1f} %")

    # ---------- 2) Memorización ----------
    r_synth, nb_synth = max_correlation(xs, x10)
    r_val, _ = max_correlation(xva, x10)
    p95_val = float(np.percentile(r_val, 95))  # los "más parecidos" que se ven entre reales distintas
    res["memorization"] = {
        "r_max_median_synthetic": float(np.median(r_synth)), "r_max_median_real_val": float(np.median(r_val)),
        "frac_synthetic_r_above_0.99": float((r_synth > 0.99).mean()),
        "frac_synthetic_above_val_percentile95": float((r_synth > p95_val).mean()),
        "percentile95_val": p95_val,
    }
    print("\n[2] Memorization (correlation with its nearest neighbor in the 10% train)")
    print(f"    median   synthetic {np.median(r_synth):.3f} | real val {np.median(r_val):.3f}")
    print(f"    synthetic with r > 0.99 (near copies): {(r_synth > 0.99).mean() * 100:.1f} %")
    print(f"    synthetic closer to train than 95% of val: {(r_synth > p95_val).mean() * 100:.1f} % (expected ~5 %)")
    if np.median(r_synth) > np.median(r_val) + 0.05 or (r_synth > 0.99).mean() > 0.02:
        print("    WARNING: the synthetic images resemble the train more than normal; the model may be copying.")

    # ---------- 3) KID por clase ----------
    res["kid"] = {}
    print("\n[3] KID (lower = closer to the real 10%)")
    for c, name_c in ((0, "normal"), (1, "pneumonia")):
        if (ys == c).sum() < 5:
            continue
        k_synth = kid(f_s[ys == c], f_10[y10 == c])
        k_ref = kid(f_va[yva == c], f_10[y10 == c])  # referencia: val real vs train real
        res["kid"][name_c] = {"synthetic_vs_train10": k_synth, "real_val_vs_train10_(reference)": k_ref}
        print(f"    {name_c:9s}: synthetic {k_synth[0]:.4f} ± {k_synth[1]:.4f} | real val (reference) {k_ref[0]:.4f} ± {k_ref[1]:.4f}")
    print("    (the verifier was trained on the 10% train, so the reference KID is somewhat optimistic)")

    # ---------- 4) Precisión / recall de distribuciones ----------
    res["precision_recall"] = {}
    print(f"\n[4] Distribution precision / recall (k = {args.pr_k}, same feature space as KID)")
    print("    precision = fidelity (synthetic inside the real support); recall = coverage / diversity")
    for c, name_c in ((0, "normal"), (1, "pneumonia")):
        if (ys == c).sum() < 5:
            continue
        pr_s = precision_recall(f_10[y10 == c], f_s[ys == c], k=args.pr_k)
        pr_ref = precision_recall(f_10[y10 == c], f_va[yva == c], k=args.pr_k)
        res["precision_recall"][name_c] = {"synthetic_vs_train10": pr_s, "real_val_vs_train10_(reference)": pr_ref}
        print(f"    {name_c:9s}: synthetic P {pr_s['precision'][0]:.3f} / R {pr_s['recall'][0]:.3f} "
              f"| real val (reference) P {pr_ref['precision'][0]:.3f} / R {pr_ref['recall'][0]:.3f} "
              f"(sets of {pr_s['n_per_set']} / {pr_ref['n_per_set']} images)")

    # ---------- 5) Incertidumbre por imagen (para la Fase 6) ----------
    p_c = np.clip(p_s, 1e-6, 1 - 1e-6)
    entropy = -(p_c * np.log(p_c) + (1 - p_c) * np.log(1 - p_c))
    margin = np.abs(2 * p_s - 1)
    if model.drop.p == 0:
        print("\nWARNING: the verifier was trained with dropout 0, so MC-dropout has nothing to vary (epistemic = 0).")
    set_seed(args.seed)
    mc_s, mc_va = predict_mc(model, xs, dev, args.mc_passes), predict_mc(model, xva, dev, args.mc_passes)
    res["uncertainty"] = {"mc_passes": int(args.mc_passes), "dropout_p": float(model.drop.p),
                          "synthetic": uncertainty_summary(mc_s, ys),
                          "real_val_(reference)": uncertainty_summary(mc_va, yva)}
    print(f"\n[5] Epistemic uncertainty (MC-dropout, {args.mc_passes} passes; mutual information, nats)")
    for c, name_c in ((0, "normal"), (1, "pneumonia")):
        s_, v_ = res["uncertainty"]["synthetic"].get(name_c), res["uncertainty"]["real_val_(reference)"].get(name_c)
        if s_ and v_:
            print(f"    {name_c:9s}: synthetic {s_['mean_mutual_info']:.4f} | real val (reference) {v_['mean_mutual_info']:.4f}")
    np.savez_compressed(VERIFIER_DIR / f"evaluation_{name}.npz", p_pneumonia=p_s, requested_label=ys,
                        entropy=entropy, margin=margin, r_max_train=r_synth, train_neighbor=nb_synth,
                        mc_p_mean=mc_s["p_mean"], mc_std=mc_s["std"], mc_entropy=mc_s["entropy"],
                        mc_expected_entropy=mc_s["expected_entropy"], mutual_info=mc_s["mutual_info"])
    with open(VERIFIER_DIR / f"evaluation_{name}.json", "w") as f:
        json.dump(res, f, indent=2)

    # ---------- Figuras ----------
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(r_val, bins=30, alpha=0.6, density=True, label="real val → train 10%")
    ax[0].hist(r_synth, bins=30, alpha=0.6, density=True, label="synthetic → train 10%")
    ax[0].set_xlabel("correlation with the most similar train image")
    ax[0].set_title("Memorization")
    ax[0].legend(fontsize=8)
    ax[1].hist(p_s[ys == 0], bins=20, alpha=0.6, label="requested normal")
    ax[1].hist(p_s[ys == 1], bins=20, alpha=0.6, label="requested pneumonia")
    ax[1].set_xlabel("P(pneumonia) according to the verifier")
    ax[1].set_title("Does it agree with the requested class?")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(VERIFIER_DIR / f"evaluation_{name}_histograms.png", dpi=130)
    plt.close(fig)

    # las 6 sintéticas más parecidas al train, con su vecina
    top = np.argsort(-r_synth)[:6]
    fig, ax = plt.subplots(2, 6, figsize=(10, 3.6))
    for j, i in enumerate(top):
        ax[0, j].imshow(xs[i], cmap="gray", vmin=0, vmax=255)
        ax[0, j].set_title(f"r={r_synth[i]:.3f}", fontsize=8)
        ax[1, j].imshow(x10[nb_synth[i]], cmap="gray", vmin=0, vmax=255)
        ax[0, j].axis("off")
        ax[1, j].axis("off")
    ax[0, 0].text(-0.1, 0.5, "synthetic", transform=ax[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    ax[1, 0].text(-0.1, 0.5, "closest\nin train", transform=ax[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(VERIFIER_DIR / f"evaluation_{name}_closest.png", dpi=130)
    plt.close(fig)
    print(f"\nResults saved in {VERIFIER_DIR} (evaluation_{name}.*)")


def main():
    ap = argparse.ArgumentParser(description="Verifier CNN and quality control of synthetic images")
    sub = ap.add_subparsers(dest="command", required=True)

    e = sub.add_parser("train", help="trains the verifier on the real 10%%")
    e.add_argument("--seed", type=int, default=0, help="which 10%% subsample to use")
    e.add_argument("--epochs", type=int, default=100)
    e.add_argument("--patience", type=int, default=15, help="epochs without improvement before stopping")
    e.add_argument("--batch", type=int, default=32)
    e.add_argument("--lr", type=float, default=1e-3)
    e.add_argument("--channels", type=int, default=32)
    e.add_argument("--dropout", type=float, default=0.3)
    e.add_argument("--no-aug", action="store_true", help="disables the small random shifts")
    e.set_defaults(func=train)

    v = sub.add_parser("evaluate", help="checks a file of synthetic images")
    v.add_argument("--seed", type=int, default=0, help="verifier and reference subsample")
    v.add_argument("--synthetic", required=True, help="path to the .npz generated by DDPM.py")
    v.add_argument("--mc-passes", type=int, default=30, help="MC-dropout passes for the epistemic uncertainty")
    v.add_argument("--pr-k", type=int, default=3, help="neighbors k for the distribution precision / recall")
    v.set_defaults(func=evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
