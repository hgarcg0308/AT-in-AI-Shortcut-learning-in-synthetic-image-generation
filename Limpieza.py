"""
Limpieza.py
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
    python Limpieza.py
    python Limpieza.py --umbral 0.99 --k 5 --fraccion 0.10 --semilla-base 0

Salida (en data/limpio/):
    pneumoniamnist_limpio.npz   datos limpios (train, val y test)
    submuestras.npz             índices de las submuestras del 10 %
    informe_limpieza.json       resumen con todos los números
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

from comun import (DIR_LIMPIO, RUTA_LIMPIO, RUTA_NPZ, RUTA_SUBMUESTRAS,
                   cargar_npz, correlacion_maxima, estandarizar)


# --------------------------------------------------------------------------
# Herramientas de duplicados
# --------------------------------------------------------------------------
def hashes(X: np.ndarray) -> list:
    """Una "huella" por imagen: si dos imágenes tienen la misma, son idénticas."""
    return [hashlib.sha1(x.tobytes()).hexdigest() for x in X]


def casi_duplicados_internos(X: np.ndarray, umbral: float, bloque: int = 512) -> np.ndarray:
    """
    Dentro de un mismo conjunto, marca las imágenes que son casi copia de otra
    anterior. Nos quedamos siempre con la primera y marcamos las siguientes.
    Devuelve un vector booleano: True = hay que quitarla.
    """
    Z = estandarizar(X)
    n = len(Z)
    quitar = np.zeros(n, dtype=bool)
    for ini in range(0, n, bloque):
        R = Z[ini:ini + bloque] @ Z.T  # correlación de este bloque contra todas
        for k in range(R.shape[0]):
            i = ini + k
            if quitar[i]:
                continue  # si ya la quitamos, no arrastra a otras
            posteriores = np.where(R[k, i + 1:] > umbral)[0] + i + 1
            quitar[posteriores] = True
    return quitar


def cuenta_por_clase(y: np.ndarray) -> dict:
    return {"normal": int((y == 0).sum()), "neumonia": int((y == 1).sum()), "total": int(len(y))}


# --------------------------------------------------------------------------
# Comprobación de que la submuestra no está desplazada respecto al resto
# --------------------------------------------------------------------------
def auc_dominio(x_a: np.ndarray, x_b: np.ndarray, semilla: int = 0) -> float:
    """
    Entrena un clasificador sencillo para distinguir "conjunto A" de "conjunto B".
    Si no puede (AUC ~ 0.5), los dos conjuntos vienen de la misma distribución.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def reducir(X):  # bajamos a la mitad de resolución para que sea rápido
        n, h, w = X.shape
        h2, w2 = h // 2 * 2, w // 2 * 2
        X = X[:, :h2, :w2].astype(np.float32) / 255.0
        return X.reshape(n, h2 // 2, 2, w2 // 2, 2).mean(axis=(2, 4)).reshape(n, -1)

    X = np.concatenate([reducir(x_a), reducir(x_b)])
    y = np.concatenate([np.zeros(len(x_a)), np.ones(len(x_b))])
    modelo = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=semilla)
    p = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Limpieza de datos y submuestras del 10 %")
    ap.add_argument("--umbral", type=float, default=0.99, help="correlación a partir de la cual dos imágenes son 'casi copias'")
    ap.add_argument("--k", type=int, default=5, help="número de submuestras (mínimo recomendado: 5)")
    ap.add_argument("--fraccion", type=float, default=0.10, help="fracción del train limpio que forma cada submuestra")
    ap.add_argument("--semilla-base", type=int, default=0, help="las semillas serán base, base+1, ...")
    ap.add_argument("--sin-dominio", action="store_true", help="salta el clasificador de dominio de las submuestras")
    args = ap.parse_args()

    if not RUTA_NPZ.exists():
        raise FileNotFoundError(f"No encuentro {RUTA_NPZ}")
    d = cargar_npz(RUTA_NPZ)
    trx, try_, vax, vay, tex, tey = (d["train_x"], d["train_y"], d["val_x"], d["val_y"],
                                     d["test_x"], d["test_y"])
    print(f"Imágenes de {trx.shape[1]}x{trx.shape[2]} píxeles")
    print(f"Original -> train {cuenta_por_clase(try_)}  val {cuenta_por_clase(vay)}  test {cuenta_por_clase(tey)}\n")

    informe = {"umbral_casi_duplicado": args.umbral,
               "original": {"train": cuenta_por_clase(try_), "val": cuenta_por_clase(vay), "test": cuenta_por_clase(tey)}}

    # ------------------------------------------------------------------
    # 1) Copias exactas dentro del train
    # ------------------------------------------------------------------
    h_train = hashes(trx)
    grupos = defaultdict(list)
    for i, h in enumerate(h_train):
        grupos[h].append(i)

    quitar_tr = np.zeros(len(trx), dtype=bool)
    n_exactos, n_ambiguos = 0, 0
    for idx in grupos.values():
        if len(idx) == 1:
            continue
        if len(set(try_[idx])) > 1:
            # misma imagen con etiquetas distintas: no sabemos cuál es correcta
            quitar_tr[idx] = True
            n_ambiguos += len(idx)
        else:
            quitar_tr[idx[1:]] = True  # nos quedamos con la primera
            n_exactos += len(idx) - 1
    print(f"[train] copias exactas eliminadas: {n_exactos}")
    print(f"[train] imágenes con etiquetas contradictorias eliminadas: {n_ambiguos}")

    # ------------------------------------------------------------------
    # 2) Casi-duplicados dentro del train (sobre lo que queda)
    # ------------------------------------------------------------------
    idx_tr = np.where(~quitar_tr)[0]
    casi_tr = casi_duplicados_internos(trx[idx_tr], args.umbral)
    n_casi_tr = int(casi_tr.sum())
    quitar_tr[idx_tr[casi_tr]] = True
    print(f"[train] casi-duplicados eliminados (r > {args.umbral}): {n_casi_tr}")

    tr_final = np.where(~quitar_tr)[0]
    trx_l, try_l = trx[tr_final], try_[tr_final]

    # ------------------------------------------------------------------
    # 3) Val: fuera todo lo que esté en el train, y las copias internas de val
    # ------------------------------------------------------------------
    set_train = set(h_train)  # huellas del train ORIGINAL (cualquier copia cuenta)
    quitar_va = np.zeros(len(vax), dtype=bool)
    vistos = set()
    n_val_en_train, n_val_internos = 0, 0
    for i, h in enumerate(hashes(vax)):
        if h in set_train:
            quitar_va[i] = True
            n_val_en_train += 1
        elif h in vistos:
            quitar_va[i] = True
            n_val_internos += 1
        vistos.add(h)
    print(f"[val]   exactas que estaban en train: {n_val_en_train}")
    print(f"[val]   copias exactas internas: {n_val_internos}")

    # casi-duplicados de val contra el train limpio
    idx_va = np.where(~quitar_va)[0]
    r_va, _ = correlacion_maxima(vax[idx_va], trx_l)
    casi_va_tr = r_va > args.umbral
    quitar_va[idx_va[casi_va_tr]] = True
    print(f"[val]   casi-duplicados contra train: {int(casi_va_tr.sum())}")

    # casi-duplicados internos de val
    idx_va = np.where(~quitar_va)[0]
    casi_va_int = casi_duplicados_internos(vax[idx_va], args.umbral)
    quitar_va[idx_va[casi_va_int]] = True
    print(f"[val]   casi-duplicados internos: {int(casi_va_int.sum())}")

    va_final = np.where(~quitar_va)[0]
    vax_l, vay_l = vax[va_final], vay[va_final]

    # ------------------------------------------------------------------
    # 4) Test: no se toca, solo se mide el parecido con el train limpio
    # ------------------------------------------------------------------
    set_train_limpio = set(hashes(trx_l))
    test_exactos = sum(h in set_train_limpio for h in hashes(tex))
    r_te, _ = correlacion_maxima(tex, trx_l)
    print(f"[test]  NO se modifica. Exactas en train: {test_exactos} | "
          f"r máx. test->train: {r_te.max():.3f} | con r > {args.umbral}: {int((r_te > args.umbral).sum())}")

    # ------------------------------------------------------------------
    # Resumen y guardado
    # ------------------------------------------------------------------
    def prior(y):
        return float((y == 1).mean() * 100) if len(y) else float("nan")

    print(f"\nLimpio   -> train {cuenta_por_clase(try_l)}  val {cuenta_por_clase(vay_l)}  test {cuenta_por_clase(tey)}")
    print(f"% neumonía: train {prior(try_l):.1f} | val {prior(vay_l):.1f} | test {prior(tey):.1f}")
    if cuenta_por_clase(vay_l)["normal"] < 50:
        print("AVISO: quedan pocas imágenes normales en val; el early stopping será ruidoso.")

    informe.update({
        "train": {"copias_exactas": n_exactos, "etiquetas_contradictorias": n_ambiguos,
                  "casi_duplicados": n_casi_tr},
        "val": {"exactas_en_train": n_val_en_train, "copias_exactas_internas": n_val_internos,
                "casi_duplicados_contra_train": int(casi_va_tr.sum()),
                "casi_duplicados_internos": int(casi_va_int.sum())},
        "test": {"exactas_en_train": int(test_exactos), "r_max_contra_train": float(r_te.max()),
                 "n_con_r_mayor_umbral": int((r_te > args.umbral).sum()), "modificado": False},
        "limpio": {"train": cuenta_por_clase(try_l), "val": cuenta_por_clase(vay_l), "test": cuenta_por_clase(tey),
                   "pct_neumonia": {"train": prior(try_l), "val": prior(vay_l), "test": prior(tey)}},
    })

    DIR_LIMPIO.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUTA_LIMPIO, train_x=trx_l, train_y=try_l, val_x=vax_l, val_y=vay_l,
                        test_x=tex, test_y=tey,
                        train_idx_original=tr_final, val_idx_original=va_final)

    # ------------------------------------------------------------------
    # 5) Submuestras del 10 % (estratificadas) con K semillas
    # ------------------------------------------------------------------
    print(f"\nCreando {args.k} submuestras del {args.fraccion * 100:.0f} % del train limpio...")
    guardar = {}
    informe["submuestras"] = []
    for k in range(args.k):
        semilla = args.semilla_base + k
        rng = np.random.default_rng(semilla)
        elegidos = []
        for c in (0, 1):
            idx_c = np.where(try_l == c)[0]
            n_c = max(1, int(round(args.fraccion * len(idx_c))))
            elegidos.append(rng.choice(idx_c, size=n_c, replace=False))
        idx10 = np.sort(np.concatenate(elegidos))
        resto = np.setdiff1d(np.arange(len(try_l)), idx10)
        guardar[f"idx_10pct_semilla{semilla}"] = idx10
        guardar[f"idx_resto_semilla{semilla}"] = resto

        fila = {"semilla": semilla, "n_10pct": int(len(idx10)), "clases": cuenta_por_clase(try_l[idx10])}
        # 6) ¿Se ha desplazado la submuestra respecto al resto?
        if not args.sin_dominio:
            fila["auc_dominio_vs_resto"] = round(auc_dominio(trx_l[idx10], trx_l[resto], semilla), 3)
        informe["submuestras"].append(fila)
        extra = f" | AUC dominio vs resto = {fila['auc_dominio_vs_resto']}" if "auc_dominio_vs_resto" in fila else ""
        print(f"  semilla {semilla}: {fila['n_10pct']} imágenes {fila['clases']}{extra}")

    np.savez_compressed(RUTA_SUBMUESTRAS, **guardar)
    with open(DIR_LIMPIO / "informe_limpieza.json", "w", encoding="utf-8") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)

    print(f"\nGuardado en {DIR_LIMPIO}")
    print("  - pneumoniamnist_limpio.npz\n  - submuestras.npz\n  - informe_limpieza.json")


if __name__ == "__main__":
    main()
