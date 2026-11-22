"""
Shift.py
========
Fase 1.2 y 1.3 del flujo de trabajo: caracterización del desplazamiento (dataset shift)
entre el TRAIN limpio y el TEST oficial. No entrena ningún generador ni verificador y
no depende de la semilla del 10 %: se ejecuta UNA vez por proyecto, después de Cleaning.py.

Qué mide, en orden:
  1. Desplazamiento de la prevalencia (prior / label shift): % de neumonía en train,
     val y test, con intervalo del 95 %, la diferencia test - train y los pesos de
     importancia p_test(y) / p_train(y).
  2. Clasificador de dominio train vs test (covariate shift): se entrena un clasificador
     para distinguir imágenes de train de imágenes de test, por clase y por región:
        - full   : imagen completa
        - edge   : solo el marco exterior de --k-center píxeles
        - center : solo el interior (sin ese marco)
     AUC ~ 0.5 = indistinguibles. Como referencia se calcula SIEMPRE:
        - un AUC "nulo": train partido al azar en dos mitades (con los mismos tamaños
          que train/test). Si el AUC train-vs-test no supera claramente al nulo, no hay
          desplazamiento detectable.
        - train vs val (solo imagen completa): cuánto se parece la validación al train.
     Un AUC alto solo en una clase indica desplazamiento condicionado a la clase
     (class-conditional shift).
  3. Test de Kolmogorov-Smirnov por estadísticos de imagen (brillo, contraste, brillo
     del borde, brillo del centro), por clase. Se reporta el estadístico D con IC por
     bootstrap y NO el p-valor (con n grande todo sale "significativo"). Como suelo de
     ruido se da el percentil 95 de D bajo permutación de etiquetas de dominio: un D por
     debajo de ese suelo es compatible con ruido de muestreo.
  4. Mapa de diferencias test - train (media de píxeles) y su perfil por distancia al
     borde, por clase, para ver si el desplazamiento se concentra en el marco.

Uso:
    python Shift.py
    python Shift.py --bootstrap 1000 --k-edge 2 --k-center 4

Salida (en outputs/shift/):
    shift.json                  todos los números
    shift_domain.csv            clasificador de dominio (una fila por grupo y región)
    shift_ks.csv                KS por clase y estadístico
    shift_difference_map.png    mapas test - train y perfil por distancia al borde
    shift_difference_maps.npz   los mapas de diferencias en bruto
"""

import os
# En Windows + conda, NumPy y PyTorch a veces cargan cada uno su copia de OpenMP y
# Python se cierra con "OMP: Error #15". Esta línea lo evita. Debe ir ANTES de importar nada más.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CLASS_NAMES, OUTPUT_DIR, load_clean

try:  # mismos colores que el resto de figuras del proyecto
    from Probes import BACKGROUND, BLUE, GRID, INK, INK_SOFT
except Exception:
    BACKGROUND, BLUE, GRID, INK, INK_SOFT = "#ffffff", "#2a6fdb", "#e3e3e3", "#222222", "#6b6b6b"
ORANGE = "#d9822b"

SHIFT_DIR = OUTPUT_DIR / "shift"
GROUPS = ("all", "normal", "pneumonia")
REGIONS = ("full", "edge", "center")


# ==========================================================================
# Utilidades de imagen
# ==========================================================================
def edge_mask(shape, k: int) -> np.ndarray:
    """Máscara booleana del marco exterior de k píxeles de grosor."""
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    m[:k, :] = m[h - k:, :] = True
    m[:, :k] = m[:, w - k:] = True
    return m


def split_groups(x: np.ndarray, y: np.ndarray) -> dict:
    return {"all": x, "normal": x[y == 0], "pneumonia": x[y == 1]}


def region_pixels(X: np.ndarray, region: str, k: int) -> np.ndarray:
    """Píxeles (en 0..1) de la región pedida, aplanados: (N, n_píxeles)."""
    Xf = X.astype(np.float32) / 255.0
    n, h, w = Xf.shape
    edge = edge_mask((h, w), k)
    if region == "full":
        return Xf.reshape(n, -1)
    return Xf[:, edge] if region == "edge" else Xf[:, ~edge]


def image_features(X: np.ndarray, k_edge: int, k_center: int) -> dict:
    """Estadísticos por imagen sobre los que se aplica el KS."""
    Xf = X.astype(np.float32) / 255.0
    shape = Xf.shape[1:]
    return {
        "brightness (mean)": Xf.mean(axis=(1, 2)),
        "contrast (std)": Xf.std(axis=(1, 2)),
        f"edge brightness ({k_edge} px)": Xf[:, edge_mask(shape, k_edge)].mean(axis=1),
        f"center brightness (without {k_center} px edge)": Xf[:, ~edge_mask(shape, k_center)].mean(axis=1),
    }


# ==========================================================================
# 1. Desplazamiento de la prevalencia
# ==========================================================================
def prior_shift(ytr, yva, yte) -> dict:
    out, p, se = {}, {}, {}
    for name, y in (("train", ytr), ("val", yva), ("test", yte)):
        n = len(y)
        p[name] = float((y == 1).mean())
        se[name] = float(np.sqrt(p[name] * (1 - p[name]) / n))
        out[name] = {"n": int(n), "pct_pneumonia": 100 * p[name],
                     "ci_low": 100 * (p[name] - 1.96 * se[name]),
                     "ci_high": 100 * (p[name] + 1.96 * se[name])}
    diff = p["test"] - p["train"]
    h = 1.96 * np.sqrt(se["train"] ** 2 + se["test"] ** 2)
    out["diff_test_minus_train_pp"] = {"value": 100 * diff, "ci_low": 100 * (diff - h), "ci_high": 100 * (diff + h)}
    out["importance_weights_test_over_train"] = {"normal": (1 - p["test"]) / (1 - p["train"]),
                                                 "pneumonia": p["test"] / p["train"]}
    return out


# ==========================================================================
# 2. Clasificador de dominio
# ==========================================================================
def domain_oof(A: np.ndarray, B: np.ndarray, seed: int, folds: int = 5):
    """
    Clasificador sencillo "A (0) vs B (1)" con validación cruzada estratificada.
    Devuelve las etiquetas de dominio y la probabilidad fuera de muestra de cada imagen.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate([A, B])
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced"))
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return y, p


def auc_with_ci(y, p, B: int, rng) -> tuple:
    """AUC y su intervalo del 95 % por bootstrap estratificado."""
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, p))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    boots = np.empty(B)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        boots[b] = roc_auc_score(y[idx], p[idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, float(lo), float(hi)


def domain_fit(A_img, B_img, region, k, seed, B, rng) -> dict:
    y, p = domain_oof(region_pixels(A_img, region, k), region_pixels(B_img, region, k), seed)
    auc, lo, hi = auc_with_ci(y, p, B, rng)
    return {"n_a": int(len(A_img)), "n_b": int(len(B_img)), "auc": auc, "ci_low": lo, "ci_high": hi}


def run_domain(tr, va, te, k_center, seed, B, rng) -> list:
    rows = []
    for g in GROUPS:
        for region in REGIONS:
            real = domain_fit(tr[g], te[g], region, k_center, seed, B, rng)
            # nulo: pseudo-test al azar del propio train, con el mismo tamaño que el test real
            perm = rng.permutation(len(tr[g]))
            n_b = len(te[g])
            null = domain_fit(tr[g][perm[n_b:]], tr[g][perm[:n_b]], region, k_center, seed, B, rng)
            row = {"group": g, "region": region, "n_train": real["n_a"], "n_test": real["n_b"],
                   "auc_train_vs_test": real["auc"], "ci_low": real["ci_low"], "ci_high": real["ci_high"],
                   "auc_null": null["auc"], "null_ci_low": null["ci_low"], "null_ci_high": null["ci_high"],
                   "auc_train_vs_val_ref": None}
            if region == "full" and len(va[g]) >= 20:
                row["auc_train_vs_val_ref"] = domain_fit(tr[g], va[g], region, k_center, seed, B, rng)["auc"]
            row["distinguishable_from_null"] = bool(row["ci_low"] > row["null_ci_high"])
            rows.append(row)
            ref = "" if row["auc_train_vs_val_ref"] is None else f" | train vs val {row['auc_train_vs_val_ref']:.3f}"
            print(f"  [{g:<9s} {region:<6s}] train vs test AUC {row['auc_train_vs_test']:.3f} "
                  f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | null {row['auc_null']:.3f}{ref}")
    return rows


# ==========================================================================
# 3. Kolmogorov-Smirnov
# ==========================================================================
def ks_stat(a, b) -> float:
    """D = sup |F_a - F_b|, calculado a mano (evita el p-valor exacto de scipy, que es lento)."""
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    fa = np.searchsorted(a, allv, side="right") / len(a)
    fb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def ks_row(a, b, B, rng) -> dict:
    """
    D con IC del 95 % por bootstrap y suelo de ruido por permutación.
    Ojo: D >= 0 siempre, así que el IC bootstrap tiende a quedar algo por encima del valor
    real cuando D es pequeño. Por eso se da también el suelo de ruido (p95 de D si train y
    test fueran la misma distribución).
    """
    d = ks_stat(a, b)
    na, nb = len(a), len(b)
    pool = np.concatenate([a, b])
    boots, floor = np.empty(B), np.empty(B)
    for i in range(B):
        boots[i] = ks_stat(rng.choice(a, na), rng.choice(b, nb))
        perm = rng.permutation(pool)
        floor[i] = ks_stat(perm[:na], perm[na:])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"D": d, "ci_low": float(lo), "ci_high": float(hi),
            "noise_floor_p95": float(np.percentile(floor, 95)),
            "median_test_minus_train": float(np.median(b) - np.median(a))}


def run_ks(tr, te, k_edge, k_center, B, rng) -> list:
    rows = []
    for g in GROUPS:
        f_tr, f_te = image_features(tr[g], k_edge, k_center), image_features(te[g], k_edge, k_center)
        for name in f_tr:
            r = ks_row(f_tr[name], f_te[name], B, rng)
            r.update({"group": g, "feature": name, "n_train": int(len(tr[g])), "n_test": int(len(te[g])),
                      "above_noise_floor": bool(r["D"] > r["noise_floor_p95"])})
            rows.append(r)
            print(f"  [{g:<9s}] {name:<38s} D = {r['D']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
                  f"(noise floor {r['noise_floor_p95']:.3f})  median diff {r['median_test_minus_train']:+.3f}")
    return rows


# ==========================================================================
# 4. Mapa de diferencias
# ==========================================================================
def ring_profile(diff: np.ndarray) -> np.ndarray:
    """Media de la diferencia en cada anillo, desde el borde (0) hacia el centro."""
    h, w = diff.shape
    ii, jj = np.indices((h, w))
    ring = np.minimum.reduce([ii, jj, h - 1 - ii, w - 1 - jj])
    return np.array([diff[ring == r].mean() for r in range(int(ring.max()) + 1)])


def difference_figure(diffs: dict, path):
    vmax = max(float(np.abs(d).max()) for d in diffs.values()) or 1e-6
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9), facecolor=BACKGROUND, constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.35]})
    for ax, (g, d) in zip(axes[:3], diffs.items()):
        im = ax.imshow(d, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{g}: test − train", color=INK, fontsize=10, loc="left")
        ax.axis("off")
    cb = fig.colorbar(im, ax=list(axes[:3]), fraction=0.03, pad=0.02)
    cb.set_label("mean intensity difference (0–1)", color=INK_SOFT, fontsize=8)
    cb.ax.tick_params(colors=INK_SOFT, labelsize=8)

    ax = axes[3]
    ax.set_facecolor(BACKGROUND)
    styles = {"normal": (BLUE, "-"), "pneumonia": (ORANGE, "-"), "all": (INK_SOFT, "--")}
    for g, d in diffs.items():
        prof = ring_profile(d)
        color, ls = styles[g]
        ax.plot(range(len(prof)), prof, ls, color=color, linewidth=2, marker="o", markersize=4, label=g)
    ax.axhline(0, color=INK_SOFT, linewidth=1)
    ax.set_xlabel("distance to the border (px, 0 = outermost ring)", color=INK_SOFT, fontsize=9)
    ax.set_ylabel("mean difference test − train", color=INK_SOFT, fontsize=9)
    ax.set_title("Profile by distance to the border", color=INK, fontsize=10, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT)
    fig.savefig(path, dpi=140, facecolor=BACKGROUND)
    plt.close(fig)


# ==========================================================================
# Utilidades de salida
# ==========================================================================
def save_csv(rows, path):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


# ==========================================================================
# Programa principal
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Dataset shift between the clean train and the official test")
    ap.add_argument("--bootstrap", type=int, default=1000, help="resamples for the intervals (default 1000)")
    ap.add_argument("--k-edge", type=int, default=2, help="border thickness (px) of the 'edge brightness' statistic")
    ap.add_argument("--k-center", type=int, default=4,
                    help="border thickness (px) that splits the image into 'edge' and 'center' regions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = load_clean()
    trx, try_, vax, vay, tex, tey = (d["train_x"], d["train_y"], d["val_x"], d["val_y"], d["test_x"], d["test_y"])
    rng = np.random.default_rng(args.seed)
    tr, va, te = split_groups(trx, try_), split_groups(vax, vay), split_groups(tex, tey)
    print(f"Clean train {len(trx)} | val {len(vax)} | test {len(tex)} | images {trx.shape[1]}x{trx.shape[2]}")

    print("\n[1] Prior (label) shift")
    prior = prior_shift(try_, vay, tey)
    for k in ("train", "val", "test"):
        p = prior[k]
        print(f"  {k:<5s} n = {p['n']:<5d} pneumonia {p['pct_pneumonia']:.1f} % [{p['ci_low']:.1f}, {p['ci_high']:.1f}]")
    dp = prior["diff_test_minus_train_pp"]
    print(f"  test − train: {dp['value']:+.1f} pp [{dp['ci_low']:+.1f}, {dp['ci_high']:+.1f}]")

    print("\n[2] Domain classifier, train vs test (AUC 0.5 = indistinguishable)")
    domain = run_domain(tr, va, te, args.k_center, args.seed, args.bootstrap, rng)

    print("\n[3] Kolmogorov-Smirnov, train vs test")
    ks = run_ks(tr, te, args.k_edge, args.k_center, args.bootstrap, rng)

    print("\n[4] Difference map test − train")
    SHIFT_DIR.mkdir(parents=True, exist_ok=True)
    diffs = {g: (te[g].astype(np.float32) / 255.0).mean(axis=0) - (tr[g].astype(np.float32) / 255.0).mean(axis=0)
             for g in GROUPS}
    difference_figure(diffs, SHIFT_DIR / "shift_difference_map.png")
    np.savez_compressed(SHIFT_DIR / "shift_difference_maps.npz", **diffs)

    report = {"settings": vars(args), "classes": CLASS_NAMES, "prior_shift": prior, "domain": domain, "ks": ks,
              "difference_profile_by_ring": {g: ring_profile(m).tolist() for g, m in diffs.items()}}
    with open(SHIFT_DIR / "shift.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    save_csv(domain, SHIFT_DIR / "shift_domain.csv")
    save_csv(ks, SHIFT_DIR / "shift_ks.csv")

    print(f"\nSaved in {SHIFT_DIR}")
    print("  - shift.json\n  - shift_domain.csv\n  - shift_ks.csv\n  - shift_difference_map.png\n  - shift_difference_maps.npz")


if __name__ == "__main__":
    main()
