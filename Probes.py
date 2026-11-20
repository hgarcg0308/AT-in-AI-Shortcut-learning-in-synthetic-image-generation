"""
Probes.py
=========
Fase 3.2 (H1) y Fase 7 (H2) del flujo de trabajo: sondas de atajo.

Idea general, sin tecnicismos:
  Una "sonda" es un clasificador muy simple al que solo le enseñamos una parte de la
  imagen (solo el brillo, solo el borde, solo el centro...). Si con esa parte tan
  pobre ya acierta la clase, es que esa parte "chiva" la respuesta: es un atajo que
  un modelo grande puede aprovechar en lugar de mirar los pulmones.

  H1 (datos reales): ¿hay atajos? Se entrena cada sonda con el 10 % real y se mide en
     la validación limpia y en el test oficial. Si el AUC cae mucho de val a test, el
     atajo no funciona igual en el test desplazado.
  H2 (datos sintéticas): ¿el DDPM los amplifica o los mitiga? Se repiten las sondas
     sobre las imágenes generadas y se comparan con las reales. Para que la
     comparación sea justa se mide lo mismo dentro de cada conjunto (validación
     cruzada), y también se mira si el atajo "viaja" de real a sintético y al revés.

  Además hay un diagnóstico que conecta el atajo con el fallo real: ¿los falsos
  positivos en las radiografías normales del test siguen al brillo del borde?
  (necesita el verificador ya entrenado; si no está, se salta).

Uso:
    # Solo H1 (datos reales)
    python Probes.py --seed 0

    # H1 + H2 (con un archivo de sintéticas generado con DDPM.py)
    python Probes.py --seed 0 --synthetic outputs/ddpm/synthetic_seed0_w2.0.npz

    # Versión rápida para una primera prueba
    python Probes.py --seed 0 --synthetic outputs/ddpm/synthetic_seed0_w2.0.npz --bootstrap 200

Salidas en outputs/probes/: tablas (.csv y .json) y figuras (.png).
"""

import os
# En Windows + conda, NumPy y PyTorch a veces cargan cada uno su copia de OpenMP y
# Python se cierra con "OMP: Error #15". Esta línea lo evita. Debe ir ANTES de importar nada más.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUTPUT_DIR, load_clean, load_subsample, pick_device

PROBES_DIR = OUTPUT_DIR / "probes"

# Colores de la paleta validada (huecos categóricos 1 y 2) y tintas de texto
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_SOFT, GRID, BACKGROUND = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


# ==========================================================================
# 1. Zonas de la imagen y características de cada sonda
# ==========================================================================
def edge_mask(size: int, k: int) -> np.ndarray:
    """Marco de k píxeles alrededor de la imagen (True = pertenece al marco)."""
    m = np.zeros((size, size), dtype=bool)
    m[:k, :] = m[-k:, :] = m[:, :k] = m[:, -k:] = True
    return m


def pixels(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Los píxeles de una zona, normalizados a [0, 1], uno por columna."""
    return X[:, mask].astype(np.float32) / 255.0


def statistics(X: np.ndarray) -> np.ndarray:
    """Solo dos números por imagen: brillo medio y contraste (desviación)."""
    f = X.reshape(len(X), -1).astype(np.float32) / 255.0
    return np.stack([f.mean(axis=1), f.std(axis=1)], axis=1)


def edge_brightness(X: np.ndarray, k: int) -> np.ndarray:
    """Brillo medio del marco de k píxeles (un único número por imagen)."""
    return pixels(X, edge_mask(X.shape[1], k)).mean(axis=1)


def define_probes(size: int, edge_ks, k_center: int, k_edge_brightness: int) -> dict:
    """Diccionario nombre -> función que saca las características de esa sonda."""
    probes = {"brightness + contrast": statistics}
    probes[f"mean edge brightness ({k_edge_brightness} px)"] = (
        lambda X, k=k_edge_brightness: edge_brightness(X, k)[:, None])
    for k in edge_ks:
        probes[f"edge only ({k} px)"] = lambda X, m=edge_mask(size, k): pixels(X, m)
    probes[f"center only (without {k_center} px edge)"] = (
        lambda X, m=~edge_mask(size, k_center): pixels(X, m))
    probes["full image"] = lambda X: X.reshape(len(X), -1).astype(np.float32) / 255.0
    return probes


# ==========================================================================
# 2. Clasificador de la sonda y medidas con intervalo de confianza
# ==========================================================================
def make_model(C: float):
    """Regresión logística sencilla (a propósito: la sonda no debe ser potente)."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def auc_with_ci(y, p, B: int, rng) -> tuple:
    """AUC y su intervalo del 95 % por bootstrap (remuestreando las imágenes)."""
    auc = float(roc_auc_score(y, p))
    n, vals = len(y), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return auc, float(lo), float(hi)


def cv_prediction(F, y, seed: int, C: float):
    """Probabilidades 'fuera de pliegue': cada imagen se predice sin haberla visto."""
    k = int(min(5, np.bincount(y).min()))
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    return cross_val_predict(make_model(C), F, y, cv=cv, method="predict_proba")[:, 1]


def make_row(probe, scenario, y, p, B, rng) -> dict:
    auc, lo, hi = auc_with_ci(y, p, B, rng)
    return {"probe": probe, "scenario": scenario, "n": int(len(y)),
            "auc": round(auc, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}


SC_VAL = "real 10% → val"
SC_TEST = "real 10% → test"
SC_CV_REAL = "real 10% (cross-validation)"
SC_CV_SYNTH = "synthetic (cross-validation)"
SC_S_VAL = "synthetic → real val"
SC_S_TEST = "synthetic → real test"
SC_R_S = "real 10% → synthetic"


def evaluate_probes(probes, d, synth, args, rng) -> list:
    """Corre todas las sondas en todos los escenarios y devuelve las filas de la tabla."""
    x10, y10, xva, yva, xte, yte = d["x10"], d["y10"], d["xva"], d["yva"], d["xte"], d["yte"]
    rows = []
    for i, (name, feat) in enumerate(probes.items(), 1):
        print(f"  probe {i}/{len(probes)}: {name}")
        F10, Fva, Fte = feat(x10), feat(xva), feat(xte)
        m = make_model(args.C).fit(F10, y10)
        rows.append(make_row(name, SC_VAL, yva, m.predict_proba(Fva)[:, 1], args.bootstrap, rng))
        rows.append(make_row(name, SC_TEST, yte, m.predict_proba(Fte)[:, 1], args.bootstrap, rng))

        if synth is not None:
            xs, ys = synth
            Fs = feat(xs)
            rows.append(make_row(name, SC_CV_REAL, y10, cv_prediction(F10, y10, args.seed, args.C),
                                 args.bootstrap, rng))
            rows.append(make_row(name, SC_CV_SYNTH, ys, cv_prediction(Fs, ys, args.seed, args.C),
                                 args.bootstrap, rng))
            ms = make_model(args.C).fit(Fs, ys)
            rows.append(make_row(name, SC_S_VAL, yva, ms.predict_proba(Fva)[:, 1], args.bootstrap, rng))
            rows.append(make_row(name, SC_S_TEST, yte, ms.predict_proba(Fte)[:, 1], args.bootstrap, rng))
            rows.append(make_row(name, SC_R_S, ys, m.predict_proba(Fs)[:, 1], args.bootstrap, rng))
    return rows


def verdict(real: dict, synth: dict) -> str:
    """Compara los intervalos de la validación cruzada real vs sintética."""
    if synth["ci_low"] > real["ci_high"]:
        return "synthetic MORE predictive (possible amplification)"
    if synth["ci_high"] < real["ci_low"]:
        return "synthetic LESS predictive (possible mitigation)"
    return "no clear difference"


# ==========================================================================
# 3. Cuánto se separan las clases en brillo (descriptivo)
# ==========================================================================
def cohen_d(a, b) -> float:
    """Diferencia entre medias en unidades de desviación (0.2 pequeña, 0.8 grande)."""
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((b.mean() - a.mean()) / sp) if sp > 0 else float("nan")


def gap_table(sets: dict, k_edge: int) -> list:
    """Para cada conjunto: brillo medio de cada clase y tamaño de la separación."""
    rows = []
    for name, (X, y) in sets.items():
        measures = {"brightness (global)": X.reshape(len(X), -1).astype(np.float32).mean(axis=1) / 255.0,
                    f"brightness (edge, {k_edge} px)": edge_brightness(X, k_edge),
                    "contrast": X.reshape(len(X), -1).astype(np.float32).std(axis=1) / 255.0}
        for med, v in measures.items():
            rows.append({"set": name, "measure": med,
                         "mean_normal": round(float(v[y == 0].mean()), 4),
                         "mean_pneumonia": round(float(v[y == 1].mean()), 4),
                         "cohen_d": round(cohen_d(v[y == 0], v[y == 1]), 3)})
    return rows


# ==========================================================================
# 4. Diagnóstico: ¿los falsos positivos siguen al brillo del borde?
# ==========================================================================
def false_positive_diagnosis(seed, d, k_edge, dev) -> list:
    """Con el verificador ya entrenado, mira qué pasa con las radiografías NORMALES."""
    from Verifier import load_verifier, predict
    try:
        ver = load_verifier(seed, dev)
    except FileNotFoundError:
        print("\n[Diagnosis] Cannot find the verifier; skipping. "
              f"(Train it with: python Verifier.py train --seed {seed})")
        return []

    print(f"\n[Diagnosis] NORMAL X-rays: do the false positives follow the edge brightness?"
          f"\n              (verifier seed {seed}, threshold 0.5, edge of {k_edge} px)")
    rows = []
    for name, X, y in (("val", d["xva"], d["yva"]), ("test", d["xte"], d["yte"])):
        Xn = X[y == 0]
        p, _ = predict(ver, Xn, dev)
        eb = edge_brightness(Xn, k_edge)
        gb = Xn.reshape(len(Xn), -1).astype(np.float32).mean(axis=1) / 255.0
        rho_e, pv_e = spearmanr(p, eb)
        rho_g, pv_g = spearmanr(p, gb)
        fp = p >= 0.5
        cuts = np.quantile(eb, [1 / 3, 2 / 3])
        tercile = np.digitize(eb, cuts)  # 0 = borde oscuro, 1 = medio, 2 = borde claro
        fp_tercile = [float(fp[tercile == t].mean() * 100) if (tercile == t).any() else float("nan") for t in range(3)]
        print(f"  {name:4s} normals (n={len(Xn)}): false positives {fp.mean() * 100:5.1f} %"
              f" | rho(P, edge brightness) = {rho_e:+.2f} (p={pv_e:.1e})"
              f" | rho(P, global brightness) = {rho_g:+.2f}")
        print(f"       % false positives by edge brightness (terciles):"
              f" dark {fp_tercile[0]:.1f} % | medium {fp_tercile[1]:.1f} % | light {fp_tercile[2]:.1f} %")
        rows.append({"set": f"{name} normal", "n": int(len(Xn)), "pct_false_positives": round(float(fp.mean() * 100), 2),
                     "rho_edge_brightness": round(float(rho_e), 3), "p_edge_brightness": float(pv_e),
                     "rho_global_brightness": round(float(rho_g), 3),
                     "fp_dark_edge": round(fp_tercile[0], 2), "fp_mid_edge": round(fp_tercile[1], 2),
                     "fp_light_edge": round(fp_tercile[2], 2)})
    print("  (A strong positive correlation on test and a weak one on val would indicate that the model\n"
          "   confuses 'light edge' with pneumonia exactly where the test is shifted.)")
    return rows


# ==========================================================================
# 5. Figuras
# ==========================================================================
def dot_figure(rows, scenarios, colors, title, path, xlim=(0.4, 1.02),
               xlabel="AUC (with 95% interval)", chance=True):
    """Un punto por sonda y escenario, con su intervalo del 95 %. Línea punteada = azar."""
    probes = list(dict.fromkeys(r["probe"] for r in rows))
    n_s = len(scenarios)
    step = 0.28
    fig, ax = plt.subplots(figsize=(8.2, 0.75 * len(probes) + 1.9), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    for j, (sc, col) in enumerate(zip(scenarios, colors)):
        ys, aucs, errs = [], [], [[], []]
        for i, p in enumerate(probes):
            r = next((x for x in rows if x["probe"] == p and x["scenario"] == sc), None)
            if r is None:
                continue
            ys.append(i + (j - (n_s - 1) / 2) * step)
            aucs.append(r["auc"])
            errs[0].append(r["auc"] - r["ci_low"])
            errs[1].append(r["ci_high"] - r["auc"])
        ax.errorbar(aucs, ys, xerr=errs, fmt="o", color=col, ecolor=col, elinewidth=2, capsize=0,
                    markersize=8, markeredgecolor=BACKGROUND, markeredgewidth=2, label=sc)
    if chance:
        ax.axvline(0.5, color=INK_SOFT, linestyle="--", linewidth=1)
        ax.text(0.5, -0.78, "chance", color=INK_SOFT, fontsize=8, ha="center")
    ax.set_yticks(range(len(probes)))
    ax.set_yticklabels(probes, color=INK, fontsize=9)
    ax.set_ylim(len(probes) - 0.4, -0.9)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=30)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0)
    # leyenda encima del gráfico, para que nunca tape ningún punto
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=n_s,
              labelcolor=INK_SOFT, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=BACKGROUND)
    plt.close(fig)


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_table(rows, scenarios):
    probes = list(dict.fromkeys(r["probe"] for r in rows))
    width = max(len(p) for p in probes) + 2
    header = "".join(f"{s[:31]:>32}" for s in scenarios)
    print(f"{'probe':<{width}}{header}")
    for p in probes:
        cells = ""
        for s in scenarios:
            r = next((x for x in rows if x["probe"] == p and x["scenario"] == s), None)
            cells += f"{r['auc']:>13.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]" if r else f"{'-':>32}"
        print(f"{p:<{width}}{cells}")


# ==========================================================================
# 6. Programa principal
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Shortcut probes on real and synthetic images")
    ap.add_argument("--seed", type=int, default=0, help="which 10%% subsample to use")
    ap.add_argument("--synthetic", default=None, help="synthetic .npz file (enables the H2 part)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4], help="edge thicknesses to try (pixels)")
    ap.add_argument("--k-center", type=int, default=4, help="edge removed for the 'center only' probe")
    ap.add_argument("--k-edge-brightness", type=int, default=2, help="edge used in the mean-brightness probe and the diagnosis")
    ap.add_argument("--C", type=float, default=0.05, help="regularization strength (lower = simpler probe)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="resamples for the intervals (200 = fast)")
    ap.add_argument("--no-diagnosis", action="store_true", help="skips the false-positive diagnosis")
    args = ap.parse_args()

    PROBES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    x10, y10, _, _ = load_subsample(args.seed)
    data = load_clean()
    d = {"x10": x10, "y10": y10, "xva": data["val_x"], "yva": data["val_y"],
         "xte": data["test_x"], "yte": data["test_y"]}
    size = x10.shape[1]
    print(f"Images of {size}x{size} | real 10%: {len(x10)} | clean val: {len(d['xva'])} | test: {len(d['xte'])}")

    synth, suffix = None, ""
    if args.synthetic:
        s = np.load(args.synthetic)
        synth = (s["images"], s["labels"].astype(int))
        suffix = Path(args.synthetic).stem.replace("synthetic_", "")  # p. ej. "seed0_w2.0"
        print(f"Synthetic: {len(synth[0])} (normal {(synth[1] == 0).sum()}, pneumonia {(synth[1] == 1).sum()})")

    probes = define_probes(size, args.ks, args.k_center, args.k_edge_brightness)
    label = suffix if suffix else f"seed{args.seed}"

    # ---------- Sondas ----------
    print("\nTraining probes...")
    rows = evaluate_probes(probes, d, synth, args, rng)

    print("\n=== H1: are there shortcuts? (probes trained on the real 10%) ===")
    print_table(rows, [SC_VAL, SC_TEST])
    print("  The farther from 0.5 with a poor part of the image, the more shortcut. If it drops from val to test,\n"
          "  that shortcut does not hold on the shifted test.")

    h2_summary = []
    if synth is not None:
        print("\n=== H2: does the DDPM amplify or mitigate the shortcut? ===")
        print("Within each set (cross-validation): how much class can be read from that part?")
        print_table(rows, [SC_CV_REAL, SC_CV_SYNTH])
        for name in probes:
            r = next(x for x in rows if x["probe"] == name and x["scenario"] == SC_CV_REAL)
            sr = next(x for x in rows if x["probe"] == name and x["scenario"] == SC_CV_SYNTH)
            v = verdict(r, sr)
            h2_summary.append({"probe": name, "auc_real": r["auc"], "auc_synthetic": sr["auc"], "verdict": v})
            print(f"  {name:<42s} real {r['auc']:.3f} | synthetic {sr['auc']:.3f} -> {v}")
        print("\nTransfer (does the shortcut from one set work on the other?):")
        print_table(rows, [SC_R_S, SC_S_VAL, SC_S_TEST])
        print("  Note: there are few synthetic images and their CI is wider; 'no clear difference' is the usual\n"
              "  outcome with few images and does not mean there is no difference.")

    # ---------- Separación de clases en brillo ----------
    sets = {"real 10%": (x10, y10), "val": (d["xva"], d["yva"]), "test": (d["xte"], d["yte"])}
    if synth is not None:
        sets["synthetic"] = synth
    gaps = gap_table(sets, args.k_edge_brightness)
    print("\n=== Class separation (Cohen's d: pneumonia - normal, in standard deviations) ===")
    print(f"{'set':<12s}{'measure':<24s}{'normal':>9s}{'pneumonia':>10s}{'d':>8s}")
    for b in gaps:
        print(f"{b['set']:<12s}{b['measure']:<24s}{b['mean_normal']:>9.3f}{b['mean_pneumonia']:>10.3f}{b['cohen_d']:>+8.2f}")

    # ---------- Diagnóstico ----------
    diag = []
    if not args.no_diagnosis:
        diag = false_positive_diagnosis(args.seed, d, args.k_edge_brightness, pick_device())

    # ---------- Guardado ----------
    save_csv(rows, PROBES_DIR / f"probes_{label}.csv")
    save_csv(gaps, PROBES_DIR / f"gaps_{label}.csv")
    with open(PROBES_DIR / f"probes_{label}.json", "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "synthetic": args.synthetic, "probes": rows, "gaps": gaps,
                   "verdict_H2": h2_summary, "false_positive_diagnosis": diag}, f, indent=2, ensure_ascii=False)

    dot_figure(rows, [SC_VAL, SC_TEST], [BLUE, ORANGE],
               "H1: how much class can be read from each part of the real image?",
               PROBES_DIR / f"h1_probes_{label}.png")
    if synth is not None:
        dot_figure(rows, [SC_CV_REAL, SC_CV_SYNTH], [BLUE, ORANGE],
                   "H2: the same, on real and on synthetic images",
                   PROBES_DIR / f"h2_probes_{label}.png")
    print(f"\nResults saved in {PROBES_DIR}")


if __name__ == "__main__":
    main()
