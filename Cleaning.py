"""
Cleaning.py
===========
Fase 1 y Fase 2 del flujo de trabajo: auditoría y limpieza de datos + simulación
de escasez.

Qué hace, en orden:
  1. Quita del TRAIN las copias exactas (mismo contenido de píxeles). Si dos copias
     tienen etiquetas distintas, se quitan las dos porque no sabemos cuál es la buena.
  2. Quita del TRAIN los casi-duplicados (correlación de píxeles > umbral).
  3. Quita de VAL todo lo que ya esté en el train (exacto o casi-duplicado) y las
     copias internas de val. Así la validación queda realmente "limpia".
  4. El TEST no se toca (es el test oficial). Solo se mide cuánto se parece al train
     para dejarlo documentado.
  5. Crea K submuestras del 10 % del train limpio, estratificadas por clase, con
     semillas distintas. El 90 % restante queda reservado para el "techo".
  6. Comprueba que cada submuestra no se ha desplazado respecto al resto del train
     (con un clasificador de dominio: AUC cercano a 0.5 = bien).

Uso:
    python Cleaning.py
    python Cleaning.py --threshold 0.99 --k 5 --fraction 0.10 --base-seed 0

Salida (en data/clean/):
    pneumoniamnist_clean.npz    datos limpios (train, val y test)
    subsamples.npz              índices de las submuestras del 10 %
    cleaning_report.json        resumen con todos los números
"""

import os
# En Windows + conda, NumPy y PyTorch a veces cargan cada uno su copia de OpenMP y
# Python se cierra con "OMP: Error #15". Esta línea lo evita. Debe ir ANTES de importar nada más.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
from collections import defaultdict

import numpy as np

from common import (CLEAN_DIR, CLEAN_PATH, NPZ_PATH, SUBSAMPLES_PATH,
                    load_npz, max_correlation, standardize)


# --------------------------------------------------------------------------
# Herramientas de duplicados
# --------------------------------------------------------------------------
def hashes(X: np.ndarray) -> list:
    """Una "huella" por imagen: si dos imágenes tienen la misma, son idénticas."""
    return [hashlib.sha1(x.tobytes()).hexdigest() for x in X]


def internal_near_duplicates(X: np.ndarray, threshold: float, block: int = 512) -> np.ndarray:
    """
    Dentro de un mismo conjunto, marca las imágenes que son casi copia de otra
    anterior. Nos quedamos siempre con la primera y marcamos las siguientes.
    Devuelve un vector booleano: True = hay que quitarla.
    """
    Z = standardize(X)
    n = len(Z)
    drop = np.zeros(n, dtype=bool)
    for start in range(0, n, block):
        R = Z[start:start + block] @ Z.T  # correlación de este bloque contra todas
        for k in range(R.shape[0]):
            i = start + k
            if drop[i]:
                continue  # si ya la quitamos, no arrastra a otras
            later = np.where(R[k, i + 1:] > threshold)[0] + i + 1
            drop[later] = True
    return drop


def count_per_class(y: np.ndarray) -> dict:
    return {"normal": int((y == 0).sum()), "pneumonia": int((y == 1).sum()), "total": int(len(y))}


# --------------------------------------------------------------------------
# Comprobación de que la submuestra no está desplazada respecto al resto
# --------------------------------------------------------------------------
def domain_auc(x_a: np.ndarray, x_b: np.ndarray, seed: int = 0) -> float:
    """
    Entrena un clasificador sencillo para distinguir "conjunto A" de "conjunto B".
    Si no puede (AUC ~ 0.5), los dos conjuntos vienen de la misma distribución.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def reduce(X):  # bajamos a la mitad de resolución para que sea rápido
        n, h, w = X.shape
        h2, w2 = h // 2 * 2, w // 2 * 2
        X = X[:, :h2, :w2].astype(np.float32) / 255.0
        return X.reshape(n, h2 // 2, 2, w2 // 2, 2).mean(axis=(2, 4)).reshape(n, -1)

    X = np.concatenate([reduce(x_a), reduce(x_b)])
    y = np.concatenate([np.zeros(len(x_a)), np.ones(len(x_b))])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Data cleaning and 10%% subsamples")
    ap.add_argument("--threshold", type=float, default=0.99, help="correlation above which two images are 'near copies'")
    ap.add_argument("--k", type=int, default=5, help="number of subsamples (recommended minimum: 5)")
    ap.add_argument("--fraction", type=float, default=0.10, help="fraction of the clean train that forms each subsample")
    ap.add_argument("--base-seed", type=int, default=0, help="seeds will be base, base+1, ...")
    ap.add_argument("--no-domain", action="store_true", help="skips the domain classifier on the subsamples")
    args = ap.parse_args()

    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"Cannot find {NPZ_PATH}")
    d = load_npz(NPZ_PATH)
    trx, try_, vax, vay, tex, tey = (d["train_x"], d["train_y"], d["val_x"], d["val_y"],
                                     d["test_x"], d["test_y"])
    print(f"Images of {trx.shape[1]}x{trx.shape[2]} pixels")
    print(f"Original -> train {count_per_class(try_)}  val {count_per_class(vay)}  test {count_per_class(tey)}\n")

    report = {"near_duplicate_threshold": args.threshold,
              "original": {"train": count_per_class(try_), "val": count_per_class(vay), "test": count_per_class(tey)}}

    # ------------------------------------------------------------------
    # 1) Copias exactas dentro del train
    # ------------------------------------------------------------------
    h_train = hashes(trx)
    groups = defaultdict(list)
    for i, h in enumerate(h_train):
        groups[h].append(i)

    drop_tr = np.zeros(len(trx), dtype=bool)
    n_exact, n_ambiguous = 0, 0
    for idx in groups.values():
        if len(idx) == 1:
            continue
        if len(set(try_[idx])) > 1:
            # misma imagen con etiquetas distintas: no sabemos cuál es correcta
            drop_tr[idx] = True
            n_ambiguous += len(idx)
        else:
            drop_tr[idx[1:]] = True  # nos quedamos con la primera
            n_exact += len(idx) - 1
    print(f"[train] exact copies removed: {n_exact}")
    print(f"[train] images with contradictory labels removed: {n_ambiguous}")

    # ------------------------------------------------------------------
    # 2) Casi-duplicados dentro del train (sobre lo que queda)
    # ------------------------------------------------------------------
    idx_tr = np.where(~drop_tr)[0]
    near_tr = internal_near_duplicates(trx[idx_tr], args.threshold)
    n_near_tr = int(near_tr.sum())
    drop_tr[idx_tr[near_tr]] = True
    print(f"[train] near-duplicates removed (r > {args.threshold}): {n_near_tr}")

    tr_final = np.where(~drop_tr)[0]
    trx_c, try_c = trx[tr_final], try_[tr_final]

    # ------------------------------------------------------------------
    # 3) Val: fuera todo lo que esté en el train, y las copias internas de val
    # ------------------------------------------------------------------
    set_train = set(h_train)  # huellas del train ORIGINAL (cualquier copia cuenta)
    drop_va = np.zeros(len(vax), dtype=bool)
    seen = set()
    n_val_in_train, n_val_internal = 0, 0
    for i, h in enumerate(hashes(vax)):
        if h in set_train:
            drop_va[i] = True
            n_val_in_train += 1
        elif h in seen:
            drop_va[i] = True
            n_val_internal += 1
        seen.add(h)
    print(f"[val]   exact matches that were in train: {n_val_in_train}")
    print(f"[val]   internal exact copies: {n_val_internal}")

    # casi-duplicados de val contra el train limpio
    idx_va = np.where(~drop_va)[0]
    r_va, _ = max_correlation(vax[idx_va], trx_c)
    near_va_tr = r_va > args.threshold
    drop_va[idx_va[near_va_tr]] = True
    print(f"[val]   near-duplicates against train: {int(near_va_tr.sum())}")

    # casi-duplicados internos de val
    idx_va = np.where(~drop_va)[0]
    near_va_int = internal_near_duplicates(vax[idx_va], args.threshold)
    drop_va[idx_va[near_va_int]] = True
    print(f"[val]   internal near-duplicates: {int(near_va_int.sum())}")

    va_final = np.where(~drop_va)[0]
    vax_c, vay_c = vax[va_final], vay[va_final]

    # ------------------------------------------------------------------
    # 4) Test: no se toca, solo se mide el parecido con el train limpio
    # ------------------------------------------------------------------
    set_train_clean = set(hashes(trx_c))
    test_exact = sum(h in set_train_clean for h in hashes(tex))
    r_te, _ = max_correlation(tex, trx_c)
    print(f"[test]  NOT modified. Exact matches in train: {test_exact} | "
          f"max r test->train: {r_te.max():.3f} | with r > {args.threshold}: {int((r_te > args.threshold).sum())}")

    # ------------------------------------------------------------------
    # Resumen y guardado
    # ------------------------------------------------------------------
    def prior(y):
        return float((y == 1).mean() * 100) if len(y) else float("nan")

    print(f"\nClean    -> train {count_per_class(try_c)}  val {count_per_class(vay_c)}  test {count_per_class(tey)}")
    print(f"% pneumonia: train {prior(try_c):.1f} | val {prior(vay_c):.1f} | test {prior(tey):.1f}")
    if count_per_class(vay_c)["normal"] < 50:
        print("WARNING: few normal images left in val; early stopping will be noisy.")

    report.update({
        "train": {"exact_copies": n_exact, "contradictory_labels": n_ambiguous,
                  "near_duplicates": n_near_tr},
        "val": {"exact_in_train": n_val_in_train, "internal_exact_copies": n_val_internal,
                "near_duplicates_vs_train": int(near_va_tr.sum()),
                "internal_near_duplicates": int(near_va_int.sum())},
        "test": {"exact_in_train": int(test_exact), "max_r_vs_train": float(r_te.max()),
                 "n_with_r_above_threshold": int((r_te > args.threshold).sum()), "modified": False},
        "clean": {"train": count_per_class(try_c), "val": count_per_class(vay_c), "test": count_per_class(tey),
                  "pct_pneumonia": {"train": prior(try_c), "val": prior(vay_c), "test": prior(tey)}},
    })

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CLEAN_PATH, train_x=trx_c, train_y=try_c, val_x=vax_c, val_y=vay_c,
                        test_x=tex, test_y=tey,
                        train_idx_original=tr_final, val_idx_original=va_final)

    # ------------------------------------------------------------------
    # 5) Submuestras del 10 % (estratificadas) con K semillas
    # ------------------------------------------------------------------
    print(f"\nCreating {args.k} subsamples of {args.fraction * 100:.0f}% of the clean train...")
    to_save = {}
    report["subsamples"] = []
    for k in range(args.k):
        seed = args.base_seed + k
        rng = np.random.default_rng(seed)
        chosen = []
        for c in (0, 1):
            idx_c = np.where(try_c == c)[0]
            n_c = max(1, int(round(args.fraction * len(idx_c))))
            chosen.append(rng.choice(idx_c, size=n_c, replace=False))
        idx10 = np.sort(np.concatenate(chosen))
        rest = np.setdiff1d(np.arange(len(try_c)), idx10)
        to_save[f"idx_10pct_seed{seed}"] = idx10
        to_save[f"idx_rest_seed{seed}"] = rest

        row = {"seed": seed, "n_10pct": int(len(idx10)), "classes": count_per_class(try_c[idx10])}
        # 6) ¿Se ha desplazado la submuestra respecto al resto?
        if not args.no_domain:
            row["domain_auc_vs_rest"] = round(domain_auc(trx_c[idx10], trx_c[rest], seed), 3)
        report["subsamples"].append(row)
        extra = f" | domain AUC vs rest = {row['domain_auc_vs_rest']}" if "domain_auc_vs_rest" in row else ""
        print(f"  seed {seed}: {row['n_10pct']} images {row['classes']}{extra}")

    np.savez_compressed(SUBSAMPLES_PATH, **to_save)
    with open(CLEAN_DIR / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved in {CLEAN_DIR}")
    print("  - pneumoniamnist_clean.npz\n  - subsamples.npz\n  - cleaning_report.json")


if __name__ == "__main__":
    main()
