"""
Scenarios.py
============
Fases 6, 8 y 9 del flujo de trabajo: selección de sintéticas por incertidumbre,
escenarios experimentales y evaluación con intervalos.

La pregunta de fondo: si al 10 % real le añadimos imágenes sintéticas, ¿el
clasificador generaliza mejor en el test desplazado? Y, ¿da igual qué sintéticas
añadamos?

Escenarios (cada clave se puede activar o no con --scenarios):
    base     10 % real, sin nada más
    classic  10 % real + copias del propio 10 % real con rotación, escala y brillo
    e1       10 % real + sintéticas elegidas al azar
    e2a      10 % real + sintéticas "fáciles"   (el verificador está muy seguro de su clase)
    e2b      10 % real + sintéticas "medias"    (confianza intermedia)
    e2c      10 % real + sintéticas "frontera"  (el verificador duda)
    e3       10 % real + sintéticas "peores"    (el verificador NO está de acuerdo con la
                                                clase pedida; control negativo)
    e4       solo sintéticas
    ceiling  todo el train limpio (cota superior con todos los datos reales)

Controles importantes:
  * Mismo número de imágenes: base aparte, todos los escenarios con sintéticas
    añaden la misma cantidad por clase, y 'classic' añade esa misma cantidad de
    copias transformadas. Así una mejora no puede deberse solo a "más datos".
  * La validación para el early stopping es SIEMPRE la val real limpia.
  * Se evalúa en el test oficial, y por separado en normales y neumonías.
  * Cada escenario se entrena varias veces (--reps) con distinta inicialización.

Ojo: el verificador que elige las sintéticas se entrenó con las mismas ~470 reales,
así que la selección está sesgada hacia lo que ya conoce (limitación del flujo).

Uso:
    python Scenarios.py --seed 0 --synthetic outputs/ddpm/synthetic_seed0_w2.0.npz

    # Versión rápida para una primera prueba
    python Scenarios.py --seed 0 --synthetic outputs/ddpm/synthetic_seed0_w2.0.npz \\
        --reps 2 --epochs 30 --bootstrap 300 --scenarios base classic e1 e2c e3

Salidas en outputs/scenarios/: tablas (.csv, .json), probabilidades (.npz) y figuras (.png).
"""

import os
# En Windows + conda, NumPy y PyTorch a veces cargan cada uno su copia de OpenMP y
# Python se cierra con "OMP: Error #15". Esta línea lo evita. Debe ir ANTES de importar nada más.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from common import OUTPUT_DIR, load_clean, load_subsample, pick_device, set_seed
from Probes import BLUE, ORANGE, dot_figure
from Verifier import VerifierCNN, to_tensor, load_verifier, shift, predict

SCENARIOS_DIR = OUTPUT_DIR / "scenarios"

NAMES = {
    "base": "Base (10% real)",
    "classic": "Classic aug. (+transformed copies)",
    "e1": "E1 real + random synth.",
    "e2a": "E2a real + easy synth.",
    "e2b": "E2b real + medium synth.",
    "e2c": "E2c real + boundary synth.",
    "e3": "E3 real + worst synth. (control)",
    "e4": "E4 synthetic only",
    "ceiling": "Ceiling (all clean train)",
}
SELECTION_MODE = {"e1": "random", "e2a": "easy", "e2b": "medium", "e2c": "boundary", "e3": "worst"}


# ==========================================================================
# 1. Elegir sintéticas según la confianza del verificador (Fase 6)
# ==========================================================================
def confidence_in_requested_class(p_pneumonia: np.ndarray, y_requested: np.ndarray) -> np.ndarray:
    """Probabilidad que el verificador da a la clase que se pidió al generar."""
    return np.where(y_requested == 1, p_pneumonia, 1 - p_pneumonia)


def select(conf, y_requested, k_per_class: int, mode: str, rng) -> np.ndarray:
    """
    Elige k sintéticas por clase según el modo:
      random   : al azar
      easy     : las de mayor confianza en la clase pedida
      medium   : las de confianza intermedia (franja central)
      boundary : las más cercanas a 0.5 (el verificador duda)
      worst    : las de menor confianza en la clase pedida (suelen tener la etiqueta "mal")
    """
    chosen = []
    for c in (0, 1):
        ic = np.where(y_requested == c)[0]
        k = min(k_per_class, len(ic))
        cc = conf[ic]
        if mode == "random":
            sel = rng.choice(ic, k, replace=False)
        elif mode == "easy":
            sel = ic[np.argsort(-cc)[:k]]
        elif mode == "worst":
            sel = ic[np.argsort(cc)[:k]]
        elif mode == "boundary":
            sel = ic[np.argsort(np.abs(cc - 0.5))[:k]]
        elif mode == "medium":
            order = np.argsort(cc)
            start = (len(ic) - k) // 2
            sel = ic[order[start:start + k]]
        else:
            raise ValueError(mode)
        chosen.append(sel)
    return np.concatenate(chosen)


# ==========================================================================
# 2. Aumentación clásica: copias del real con rotación, escala, desplazamiento y brillo
# ==========================================================================
def classic_augment(x: np.ndarray, y: np.ndarray, n: int, rng):
    """Genera n copias transformadas de imágenes reales elegidas al azar."""
    size = x.shape[1]
    idx = rng.choice(len(x), n, replace=True)
    X = torch.from_numpy(x[idx]).float().div(255).unsqueeze(1)
    ang = torch.from_numpy(rng.uniform(-10, 10, n) * np.pi / 180).float()
    scale = torch.from_numpy(rng.uniform(0.9, 1.1, n)).float()
    tx = torch.from_numpy(rng.uniform(-2, 2, n) * 2 / size).float()
    ty = torch.from_numpy(rng.uniform(-2, 2, n) * 2 / size).float()
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = ang.cos() / scale, -ang.sin() / scale, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = ang.sin() / scale, ang.cos() / scale, ty
    grid = F.affine_grid(theta, X.shape, align_corners=False)
    X = F.grid_sample(X, grid, padding_mode="border", align_corners=False)
    # brillo y contraste
    mean = X.mean(dim=(1, 2, 3), keepdim=True)
    contrast = torch.from_numpy(rng.uniform(0.9, 1.1, n)).float().view(n, 1, 1, 1)
    brightness = torch.from_numpy(rng.uniform(-0.05, 0.05, n)).float().view(n, 1, 1, 1)
    X = ((X - mean) * contrast + mean + brightness).clamp(0, 1)
    return (X * 255).round().byte().squeeze(1).numpy(), y[idx]


# ==========================================================================
# 3. Entrenar un clasificador (mismo modelo que el verificador)
# ==========================================================================
def train_classifier(x, y, xva, yva, dev, seed, args):
    """Entrena con early stopping sobre la AUC de la val real. Devuelve el modelo ya en su mejor época."""
    set_seed(seed)
    X, Y = to_tensor(x, dev), torch.from_numpy(y).float().to(dev)
    model = VerifierCNN(args.channels, 0.3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    pos_weight = torch.tensor((y == 0).sum() / max(1, (y == 1).sum()), device=dev).float()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_loss, best_state, no_improve = -1.0, float("inf"), None, 0
    for _ in range(args.epochs):
        model.train()
        order = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), args.batch):
            idx = order[i:i + args.batch]
            if len(idx) < 2:
                continue
            xb = shift(X[idx]) if args.train_aug else X[idx]
            loss = criterion(model(xb), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        p, _ = predict(model, xva, dev)
        auc = roc_auc_score(yva, p)
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        loss_va = float(-(yva * np.log(pc) + (1 - yva) * np.log(1 - pc)).mean())
        # mejor = más AUC; si empatan, gana la menor pérdida
        if auc > best_auc + 1e-4 or (abs(auc - best_auc) <= 1e-4 and loss_va < best_loss):
            best_auc, best_loss, no_improve = max(auc, best_auc), loss_va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break
    model.load_state_dict(best_state)
    return model


# ==========================================================================
# 4. Métricas e intervalos (Fase 9)
# ==========================================================================
def ece(y, p, bins=10) -> float:
    """Error de calibración: cuánto se desvía la probabilidad que da el modelo de la frecuencia real."""
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def point_metrics(y, P) -> dict:
    """P: (reps, n) probabilidades. Devuelve la media sobre las réplicas de entrenamiento."""
    aucs = [roc_auc_score(y, p) for p in P]
    pred = P >= 0.5
    return {"auc": float(np.mean(aucs)), "auc_sd_reps": float(np.std(aucs)),
            "sens": float(pred[:, y == 1].mean()), "spec": float((~pred[:, y == 0]).mean()),
            "ece": float(np.mean([ece(y, p) for p in P]))}


def paired_bootstrap(y, P_by_sc: dict, B: int, rng) -> dict:
    """
    Remuestrea las imágenes del test y calcula las métricas de TODOS los escenarios con
    los mismos remuestreos. Así las diferencias entre escenarios son pareadas (misma
    muestra de test), que es más justo que comparar dos intervalos sueltos.
    """
    n, names = len(y), list(P_by_sc)
    out = {m: {s: np.full(B, np.nan) for s in names} for m in ("auc", "sens", "spec")}
    for b in range(B):
        i = rng.integers(0, n, n)
        yb = y[i]
        if len(np.unique(yb)) < 2:
            continue
        pos = yb == 1
        for s in names:
            pb = P_by_sc[s][:, i]
            pred = pb >= 0.5
            out["auc"][s][b] = np.mean([roc_auc_score(yb, p) for p in pb])
            out["sens"][s][b] = pred[:, pos].mean()
            out["spec"][s][b] = (~pred[:, ~pos]).mean()
    return out


def ci(v) -> tuple:
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return float(lo), float(hi)


# ==========================================================================
# 5. Montar los conjuntos de entrenamiento de cada escenario
# ==========================================================================
def build_scenarios(args, x10, y10, data, pool, rng):
    """Devuelve {clave: (x, y)} con el conjunto de entrenamiento de cada escenario."""
    n_real = len(x10)
    xs, ys, conf = pool if pool is not None else (None, None, None)

    k = 0
    if xs is not None:
        available = min((ys == 0).sum(), (ys == 1).sum())
        requested = args.n_synth_per_class or int(round(args.ratio * n_real / 2))
        # por defecto no usamos más de la mitad del pool por clase, para que filtrar tenga sentido
        k = min(requested, available // 2) if args.n_synth_per_class is None else min(requested, available)
        k = max(k, 1)
        print(f"Synthetic available: {available} per class | adding {k} per class ({2 * k} in total) "
              f"to the {n_real} real images")
        if k >= available:
            print("WARNING: the whole pool is used, so the filters (easy/medium/boundary/worst) "
                  "will pick the same images. Generate more synthetic images or lower --n-synth-per-class.")

    sc = {}
    for key in args.scenarios:
        if key == "base":
            sc[key] = (x10, y10)
        elif key == "classic":
            xa, ya = classic_augment(x10, y10, 2 * k, rng)
            sc[key] = (np.concatenate([x10, xa]), np.concatenate([y10, ya]))
        elif key in SELECTION_MODE:
            sel = select(conf, ys, k, SELECTION_MODE[key], rng)
            sc[key] = (np.concatenate([x10, xs[sel]]), np.concatenate([y10, ys[sel]]))
        elif key == "e4":
            kt = max(1, int(round((n_real + 2 * k) / 2)))
            sel = select(conf if conf is not None else np.zeros(len(ys)), ys, kt, "random", rng)
            sc[key] = (xs[sel], ys[sel])
        elif key == "ceiling":
            sc[key] = (data["train_x"], data["train_y"])
    return sc


# ==========================================================================
# 6. Salidas
# ==========================================================================
def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        columns = list(dict.fromkeys(k for row in rows for k in row))  # la base no tiene columnas "delta"
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Experimental scenarios with synthetic images, evaluation with intervals")
    ap.add_argument("--seed", type=int, default=0, help="10%% subsample (and verifier) to use")
    ap.add_argument("--synthetic", default=None, help=".npz file generated by DDPM.py")
    ap.add_argument("--scenarios", nargs="+", default=list(NAMES), choices=list(NAMES))
    ap.add_argument("--ratio", type=float, default=1.0, help="synthetic added = ratio x number of real (e.g. 1, 2, 5)")
    ap.add_argument("--n-synth-per-class", type=int, default=None, help="fixes how many synthetic images per class are added")
    ap.add_argument("--reps", type=int, default=3, help="trainings per scenario (different initialization)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--train-aug", action="store_true",
                    help="adds small shifts while training (default NO, so it is not mixed with 'classic')")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    needs_synth = [e for e in args.scenarios if e not in ("base", "classic", "ceiling")]
    if needs_synth and not args.synthetic:
        raise SystemExit(f"Scenarios {needs_synth} need --synthetic (file from DDPM.py)")

    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)
    dev = pick_device()
    rng = np.random.default_rng(args.seed)

    x10, y10, _, _ = load_subsample(args.seed)
    data = load_clean()
    xva, yva, xte, yte = data["val_x"], data["val_y"], data["test_x"], data["test_y"]
    print(f"Device: {dev} | real 10%: {len(x10)} | clean val: {len(xva)} | test: {len(xte)}")

    # ---------- sintéticas y confianza del verificador ----------
    pool, label = None, f"seed{args.seed}"
    if args.synthetic:
        s = np.load(args.synthetic)
        xs, ys = s["images"], s["labels"].astype(int)
        conf = None
        if any(e in SELECTION_MODE and e != "e1" for e in args.scenarios):
            ver = load_verifier(args.seed, dev)  # avisa si no está entrenado
            p_s, _ = predict(ver, xs, dev)
            conf = confidence_in_requested_class(p_s, ys)
            print(f"Verifier: mean confidence in the requested class = {conf.mean():.3f} "
                  f"| disagreement with the requested class = {(conf < 0.5).mean() * 100:.1f} %")
        pool = (xs, ys, conf)
        label = Path(args.synthetic).stem.replace("synthetic_", "") + f"_x{args.ratio:g}"

    sc = build_scenarios(args, x10, y10, data, pool, rng)

    print("\nTraining sets:")
    for c, (x, y) in sc.items():
        print(f"  {NAMES[c]:<40s} {len(x):5d} images (normal {(y == 0).sum()}, pneumonia {(y == 1).sum()})")

    # ---------- entrenar ----------
    P_test, P_val = {}, {}
    t0 = time.time()
    for c, (x, y) in sc.items():
        pt, pv = [], []
        for r in range(args.reps):
            m = train_classifier(x, y, xva, yva, dev, args.seed * 100 + r, args)
            pt.append(predict(m, xte, dev)[0])
            pv.append(predict(m, xva, dev)[0])
        P_test[NAMES[c]], P_val[NAMES[c]] = np.array(pt), np.array(pv)
        mp = point_metrics(yte, P_test[NAMES[c]])
        print(f"  [{time.time() - t0:5.0f}s] {NAMES[c]:<40s} test AUC {mp['auc']:.3f} "
              f"(± {mp['auc_sd_reps']:.3f} across replicates)")

    # ---------- métricas e intervalos ----------
    print(f"\nPaired bootstrap on the test set ({args.bootstrap} resamples)...")
    boot = paired_bootstrap(yte, P_test, args.bootstrap, rng)
    base = NAMES["base"] if "base" in args.scenarios else None

    rows = []
    for name in P_test:
        mp = point_metrics(yte, P_test[name])
        mv = point_metrics(yva, P_val[name])
        row = {"scenario": name, "n_train": int(len(sc[[k for k, v in NAMES.items() if v == name][0]][0])),
               "auc_test": round(mp["auc"], 4), "auc_test_sd_reps": round(mp["auc_sd_reps"], 4),
               "auc_test_ci_low": round(ci(boot["auc"][name])[0], 4), "auc_test_ci_high": round(ci(boot["auc"][name])[1], 4),
               "sens_test": round(mp["sens"], 4), "sens_ci_low": round(ci(boot["sens"][name])[0], 4),
               "sens_ci_high": round(ci(boot["sens"][name])[1], 4),
               "spec_test": round(mp["spec"], 4), "spec_ci_low": round(ci(boot["spec"][name])[0], 4),
               "spec_ci_high": round(ci(boot["spec"][name])[1], 4),
               "ece_test": round(mp["ece"], 4), "auc_val": round(mv["auc"], 4)}
        if base and name != base:
            for m in ("auc", "spec"):
                d = boot[m][name] - boot[m][base]
                lo, hi = ci(d)
                row[f"delta_{m}_vs_base"] = round(float(np.nanmean(d)), 4)
                row[f"delta_{m}_ci_low"], row[f"delta_{m}_ci_high"] = round(lo, 4), round(hi, 4)
        rows.append(row)

    # ---------- tabla en pantalla ----------
    print("\n=== Results on the official test set (mean of replicates; 95% CI by paired bootstrap) ===")
    print(f"{'scenario':<40s}{'test AUC [CI]':>26s}{'Δ AUC vs base [CI]':>30s}{'sens':>7s}{'spec':>7s}{'ECE':>7s}{'val AUC':>9s}")
    for r in rows:
        d = ""
        if "delta_auc_vs_base" in r:
            mark = "*" if (r["delta_auc_ci_low"] > 0 or r["delta_auc_ci_high"] < 0) else " "
            d = f"{r['delta_auc_vs_base']:+.3f} [{r['delta_auc_ci_low']:+.3f}, {r['delta_auc_ci_high']:+.3f}]{mark}"
        print(f"{r['scenario']:<40s}{r['auc_test']:>9.3f} [{r['auc_test_ci_low']:.3f}, {r['auc_test_ci_high']:.3f}]"
              f"{d:>30s}{r['sens_test']:>7.3f}{r['spec_test']:>7.3f}{r['ece_test']:>7.3f}{r['auc_val']:>9.3f}")
    print("  * = the interval of the difference does not include 0.")
    if base:
        print("\n=== Test specificity versus base (where the shift fails) ===")
        for r in rows:
            if "delta_spec_vs_base" in r:
                mark = "*" if (r["delta_spec_ci_low"] > 0 or r["delta_spec_ci_high"] < 0) else " "
                print(f"  {r['scenario']:<40s} spec {r['spec_test']:.3f} | Δ {r['delta_spec_vs_base']:+.3f} "
                      f"[{r['delta_spec_ci_low']:+.3f}, {r['delta_spec_ci_high']:+.3f}]{mark}")
    print("\nCaution when interpreting: these intervals only capture the finite test set (624 images) and the\n"
          "initialization; they come from ONE 10% subsample. To say something general, repeat with\n"
          "other seeds (--seed 1..4) and compare the differences between them. AUC differences of\n"
          "0.01-0.02 are usually indistinguishable from noise.")

    # ---------- guardado ----------
    save_csv(rows, SCENARIOS_DIR / f"scenarios_{label}.csv")
    with open(SCENARIOS_DIR / f"scenarios_{label}.json", "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "synthetic": args.synthetic, "ratio": args.ratio,
                   "reps": args.reps, "bootstrap": args.bootstrap, "results": rows}, f, indent=2, ensure_ascii=False)
    np.savez_compressed(SCENARIOS_DIR / f"probabilities_{label}.npz", y_test=yte, y_val=yva,
                        **{f"test::{k}": v for k, v in P_test.items()}, **{f"val::{k}": v for k, v in P_val.items()})

    rows_auc = [{"probe": r["scenario"], "scenario": "Test AUC", "auc": r["auc_test"],
                 "ci_low": r["auc_test_ci_low"], "ci_high": r["auc_test_ci_high"]} for r in rows]
    dot_figure(rows_auc, ["Test AUC"], [BLUE], "AUC on the official test set by scenario",
               SCENARIOS_DIR / f"auc_{label}.png")
    rows_ss = []
    for r in rows:
        rows_ss.append({"probe": r["scenario"], "scenario": "Sensitivity (pneumonia correct)", "auc": r["sens_test"],
                        "ci_low": r["sens_ci_low"], "ci_high": r["sens_ci_high"]})
        rows_ss.append({"probe": r["scenario"], "scenario": "Specificity (normal correct)", "auc": r["spec_test"],
                        "ci_low": r["spec_ci_low"], "ci_high": r["spec_ci_high"]})
    dot_figure(rows_ss, ["Sensitivity (pneumonia correct)", "Specificity (normal correct)"], [BLUE, ORANGE],
               "Per-class accuracy on test (threshold 0.5)", SCENARIOS_DIR / f"sens_spec_{label}.png",
               xlim=(0, 1.02), xlabel="Proportion correct (with 95% interval)", chance=False)
    print(f"\nResults saved in {SCENARIOS_DIR} (…{label}.*)")


if __name__ == "__main__":
    main()
