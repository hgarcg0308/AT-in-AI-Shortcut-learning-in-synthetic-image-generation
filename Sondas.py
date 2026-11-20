"""
Sondas.py
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
    python Sondas.py --semilla 0

    # H1 + H2 (con un archivo de sintéticas generado con DDPM.py)
    python Sondas.py --semilla 0 --sinteticas outputs/ddpm/sinteticas_semilla0_w2.0.npz

    # Versión rápida para una primera prueba
    python Sondas.py --semilla 0 --sinteticas outputs/ddpm/sinteticas_semilla0_w2.0.npz --bootstrap 200

Salidas en outputs/sondas/: tablas (.csv y .json) y figuras (.png).
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

from comun import DIR_SALIDAS, cargar_limpio, cargar_submuestra, elegir_dispositivo

DIR_SONDAS = DIR_SALIDAS / "sondas"

# Colores de la paleta validada (huecos categóricos 1 y 2) y tintas de texto
AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_SUAVE, REJILLA, FONDO = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


# ==========================================================================
# 1. Zonas de la imagen y características de cada sonda
# ==========================================================================
def mascara_borde(tam: int, k: int) -> np.ndarray:
    """Marco de k píxeles alrededor de la imagen (True = pertenece al marco)."""
    m = np.zeros((tam, tam), dtype=bool)
    m[:k, :] = m[-k:, :] = m[:, :k] = m[:, -k:] = True
    return m


def pixeles(X: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Los píxeles de una zona, normalizados a [0, 1], uno por columna."""
    return X[:, mascara].astype(np.float32) / 255.0


def estadisticos(X: np.ndarray) -> np.ndarray:
    """Solo dos números por imagen: brillo medio y contraste (desviación)."""
    f = X.reshape(len(X), -1).astype(np.float32) / 255.0
    return np.stack([f.mean(axis=1), f.std(axis=1)], axis=1)


def brillo_borde(X: np.ndarray, k: int) -> np.ndarray:
    """Brillo medio del marco de k píxeles (un único número por imagen)."""
    return pixeles(X, mascara_borde(X.shape[1], k)).mean(axis=1)


def definir_sondas(tam: int, ks_borde, k_centro: int, k_brillo_borde: int) -> dict:
    """Diccionario nombre -> función que saca las características de esa sonda."""
    sondas = {"brillo + contraste": estadisticos}
    sondas[f"brillo medio del borde ({k_brillo_borde} px)"] = (
        lambda X, k=k_brillo_borde: brillo_borde(X, k)[:, None])
    for k in ks_borde:
        sondas[f"solo borde ({k} px)"] = lambda X, m=mascara_borde(tam, k): pixeles(X, m)
    sondas[f"solo centro (sin borde de {k_centro} px)"] = (
        lambda X, m=~mascara_borde(tam, k_centro): pixeles(X, m))
    sondas["imagen completa"] = lambda X: X.reshape(len(X), -1).astype(np.float32) / 255.0
    return sondas


# ==========================================================================
# 2. Clasificador de la sonda y medidas con intervalo de confianza
# ==========================================================================
def modelo(C: float):
    """Regresión logística sencilla (a propósito: la sonda no debe ser potente)."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


def auc_con_ic(y, p, B: int, rng) -> tuple:
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


def prediccion_cv(F, y, semilla: int, C: float):
    """Probabilidades 'fuera de pliegue': cada imagen se predice sin haberla visto."""
    k = int(min(5, np.bincount(y).min()))
    cv = StratifiedKFold(k, shuffle=True, random_state=semilla)
    return cross_val_predict(modelo(C), F, y, cv=cv, method="predict_proba")[:, 1]


def fila(sonda, escenario, y, p, B, rng) -> dict:
    auc, lo, hi = auc_con_ic(y, p, B, rng)
    return {"sonda": sonda, "escenario": escenario, "n": int(len(y)),
            "auc": round(auc, 4), "ic_bajo": round(lo, 4), "ic_alto": round(hi, 4)}


ESC_VAL = "real 10 % → val"
ESC_TEST = "real 10 % → test"
ESC_CV_REAL = "real 10 % (validación cruzada)"
ESC_CV_SINT = "sintéticas (validación cruzada)"
ESC_S_VAL = "sintéticas → val real"
ESC_S_TEST = "sintéticas → test real"
ESC_R_S = "real 10 % → sintéticas"


def evaluar_sondas(sondas, d, sint, args, rng) -> list:
    """Corre todas las sondas en todos los escenarios y devuelve las filas de la tabla."""
    x10, y10, xva, yva, xte, yte = d["x10"], d["y10"], d["xva"], d["yva"], d["xte"], d["yte"]
    filas = []
    for i, (nombre, feat) in enumerate(sondas.items(), 1):
        print(f"  sonda {i}/{len(sondas)}: {nombre}")
        F10, Fva, Fte = feat(x10), feat(xva), feat(xte)
        m = modelo(args.C).fit(F10, y10)
        filas.append(fila(nombre, ESC_VAL, yva, m.predict_proba(Fva)[:, 1], args.bootstrap, rng))
        filas.append(fila(nombre, ESC_TEST, yte, m.predict_proba(Fte)[:, 1], args.bootstrap, rng))

        if sint is not None:
            xs, ys = sint
            Fs = feat(xs)
            filas.append(fila(nombre, ESC_CV_REAL, y10, prediccion_cv(F10, y10, args.semilla, args.C),
                              args.bootstrap, rng))
            filas.append(fila(nombre, ESC_CV_SINT, ys, prediccion_cv(Fs, ys, args.semilla, args.C),
                              args.bootstrap, rng))
            ms = modelo(args.C).fit(Fs, ys)
            filas.append(fila(nombre, ESC_S_VAL, yva, ms.predict_proba(Fva)[:, 1], args.bootstrap, rng))
            filas.append(fila(nombre, ESC_S_TEST, yte, ms.predict_proba(Fte)[:, 1], args.bootstrap, rng))
            filas.append(fila(nombre, ESC_R_S, ys, m.predict_proba(Fs)[:, 1], args.bootstrap, rng))
    return filas


def veredicto(real: dict, sint: dict) -> str:
    """Compara los intervalos de la validación cruzada real vs sintética."""
    if sint["ic_bajo"] > real["ic_alto"]:
        return "sintéticas MÁS predictivas (posible amplificación)"
    if sint["ic_alto"] < real["ic_bajo"]:
        return "sintéticas MENOS predictivas (posible mitigación)"
    return "sin diferencia clara"


# ==========================================================================
# 3. Cuánto se separan las clases en brillo (descriptivo)
# ==========================================================================
def d_cohen(a, b) -> float:
    """Diferencia entre medias en unidades de desviación (0.2 pequeña, 0.8 grande)."""
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((b.mean() - a.mean()) / sp) if sp > 0 else float("nan")


def tabla_brechas(conjuntos: dict, k_borde: int) -> list:
    """Para cada conjunto: brillo medio de cada clase y tamaño de la separación."""
    filas = []
    for nombre, (X, y) in conjuntos.items():
        medidas = {"brillo global": X.reshape(len(X), -1).astype(np.float32).mean(axis=1) / 255.0,
                   f"brillo del borde ({k_borde} px)": brillo_borde(X, k_borde),
                   "contraste": X.reshape(len(X), -1).astype(np.float32).std(axis=1) / 255.0}
        for med, v in medidas.items():
            filas.append({"conjunto": nombre, "medida": med,
                          "media_normal": round(float(v[y == 0].mean()), 4),
                          "media_neumonia": round(float(v[y == 1].mean()), 4),
                          "d_cohen": round(d_cohen(v[y == 0], v[y == 1]), 3)})
    return filas


# ==========================================================================
# 4. Diagnóstico: ¿los falsos positivos siguen al brillo del borde?
# ==========================================================================
def diagnostico_falsos_positivos(semilla, d, k_borde, dev) -> list:
    """Con el verificador ya entrenado, mira qué pasa con las radiografías NORMALES."""
    from Verificador import cargar_verificador, predecir
    try:
        ver = cargar_verificador(semilla, dev)
    except FileNotFoundError:
        print("\n[Diagnóstico] No encuentro el verificador; se salta. "
              f"(Entrénalo con: python Verificador.py entrenar --semilla {semilla})")
        return []

    print(f"\n[Diagnóstico] Radiografías NORMALES: ¿los falsos positivos siguen al brillo del borde?"
          f"\n              (verificador semilla {semilla}, umbral 0.5, borde de {k_borde} px)")
    filas = []
    for nombre, X, y in (("val", d["xva"], d["yva"]), ("test", d["xte"], d["yte"])):
        Xn = X[y == 0]
        p, _ = predecir(ver, Xn, dev)
        bb = brillo_borde(Xn, k_borde)
        bg = Xn.reshape(len(Xn), -1).astype(np.float32).mean(axis=1) / 255.0
        rho_b, pv_b = spearmanr(p, bb)
        rho_g, pv_g = spearmanr(p, bg)
        fp = p >= 0.5
        cortes = np.quantile(bb, [1 / 3, 2 / 3])
        tramo = np.digitize(bb, cortes)  # 0 = borde oscuro, 1 = medio, 2 = borde claro
        fp_tramo = [float(fp[tramo == t].mean() * 100) if (tramo == t).any() else float("nan") for t in range(3)]
        print(f"  {nombre:4s} normales (n={len(Xn)}): falsos positivos {fp.mean() * 100:5.1f} %"
              f" | rho(P, brillo borde) = {rho_b:+.2f} (p={pv_b:.1e})"
              f" | rho(P, brillo global) = {rho_g:+.2f}")
        print(f"       % falsos positivos según el brillo del borde (terciles):"
              f" oscuro {fp_tramo[0]:.1f} % | medio {fp_tramo[1]:.1f} % | claro {fp_tramo[2]:.1f} %")
        filas.append({"conjunto": f"{nombre} normal", "n": int(len(Xn)), "pct_falsos_positivos": round(float(fp.mean() * 100), 2),
                      "rho_brillo_borde": round(float(rho_b), 3), "p_brillo_borde": float(pv_b),
                      "rho_brillo_global": round(float(rho_g), 3),
                      "fp_borde_oscuro": round(fp_tramo[0], 2), "fp_borde_medio": round(fp_tramo[1], 2),
                      "fp_borde_claro": round(fp_tramo[2], 2)})
    print("  (Una correlación positiva fuerte en test y débil en val indicaría que el modelo\n"
          "   confunde 'borde claro' con neumonía justo donde el test está desplazado.)")
    return filas


# ==========================================================================
# 5. Figuras
# ==========================================================================
def figura_puntos(filas, escenarios, colores, titulo, ruta, xlim=(0.4, 1.02),
                  xlabel="AUC (con intervalo del 95 %)", azar=True):
    """Un punto por sonda y escenario, con su intervalo del 95 %. Línea punteada = azar."""
    sondas = list(dict.fromkeys(f["sonda"] for f in filas))
    n_e = len(escenarios)
    paso = 0.28
    fig, ax = plt.subplots(figsize=(8.2, 0.75 * len(sondas) + 1.9), facecolor=FONDO)
    ax.set_facecolor(FONDO)
    for j, (esc, col) in enumerate(zip(escenarios, colores)):
        ys, aucs, errs = [], [], [[], []]
        for i, s in enumerate(sondas):
            f = next((x for x in filas if x["sonda"] == s and x["escenario"] == esc), None)
            if f is None:
                continue
            ys.append(i + (j - (n_e - 1) / 2) * paso)
            aucs.append(f["auc"])
            errs[0].append(f["auc"] - f["ic_bajo"])
            errs[1].append(f["ic_alto"] - f["auc"])
        ax.errorbar(aucs, ys, xerr=errs, fmt="o", color=col, ecolor=col, elinewidth=2, capsize=0,
                    markersize=8, markeredgecolor=FONDO, markeredgewidth=2, label=esc)
    if azar:
        ax.axvline(0.5, color=TINTA_SUAVE, linestyle="--", linewidth=1)
        ax.text(0.5, -0.78, "azar", color=TINTA_SUAVE, fontsize=8, ha="center")
    ax.set_yticks(range(len(sondas)))
    ax.set_yticklabels(sondas, color=TINTA, fontsize=9)
    ax.set_ylim(len(sondas) - 0.4, -0.9)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, color=TINTA_SUAVE, fontsize=9)
    ax.set_title(titulo, color=TINTA, fontsize=11, loc="left", pad=30)
    ax.grid(axis="x", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(REJILLA)
    ax.tick_params(colors=TINTA_SUAVE, length=0)
    # leyenda encima del gráfico, para que nunca tape ningún punto
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=n_e,
              labelcolor=TINTA_SUAVE, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140, facecolor=FONDO)
    plt.close(fig)


def guardar_csv(filas, ruta):
    if not filas:
        return
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def imprimir_tabla(filas, escenarios):
    sondas = list(dict.fromkeys(f["sonda"] for f in filas))
    ancho = max(len(s) for s in sondas) + 2
    cab = "".join(f"{e[:31]:>32}" for e in escenarios)
    print(f"{'sonda':<{ancho}}{cab}")
    for s in sondas:
        celdas = ""
        for e in escenarios:
            f = next((x for x in filas if x["sonda"] == s and x["escenario"] == e), None)
            celdas += f"{f['auc']:>13.3f} [{f['ic_bajo']:.3f}, {f['ic_alto']:.3f}]" if f else f"{'-':>32}"
        print(f"{s:<{ancho}}{celdas}")


# ==========================================================================
# 6. Programa principal
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Sondas de atajo sobre imágenes reales y sintéticas")
    ap.add_argument("--semilla", type=int, default=0, help="qué submuestra del 10 %% usar")
    ap.add_argument("--sinteticas", default=None, help="archivo .npz de sintéticas (activa la parte H2)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4], help="grosores de borde a probar (píxeles)")
    ap.add_argument("--k-centro", type=int, default=4, help="borde que se quita para la sonda 'solo centro'")
    ap.add_argument("--k-brillo-borde", type=int, default=2, help="borde usado en la sonda de brillo medio y el diagnóstico")
    ap.add_argument("--C", type=float, default=0.05, help="fuerza de regularización (más bajo = sonda más simple)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="remuestreos para los intervalos (200 = rápido)")
    ap.add_argument("--sin-diagnostico", action="store_true", help="salta el diagnóstico de falsos positivos")
    args = ap.parse_args()

    DIR_SONDAS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.semilla)

    x10, y10, _, _ = cargar_submuestra(args.semilla)
    datos = cargar_limpio()
    d = {"x10": x10, "y10": y10, "xva": datos["val_x"], "yva": datos["val_y"],
         "xte": datos["test_x"], "yte": datos["test_y"]}
    tam = x10.shape[1]
    print(f"Imágenes de {tam}x{tam} | real 10 %: {len(x10)} | val limpia: {len(d['xva'])} | test: {len(d['xte'])}")

    sint, sufijo = None, ""
    if args.sinteticas:
        s = np.load(args.sinteticas)
        sint = (s["images"], s["labels"].astype(int))
        sufijo = Path(args.sinteticas).stem.replace("sinteticas_", "")  # p. ej. "semilla0_w2.0"
        print(f"Sintéticas: {len(sint[0])} (normal {(sint[1] == 0).sum()}, neumonía {(sint[1] == 1).sum()})")

    sondas = definir_sondas(tam, args.ks, args.k_centro, args.k_brillo_borde)
    etiqueta = sufijo if sufijo else f"semilla{args.semilla}"

    # ---------- Sondas ----------
    print("\nEntrenando sondas...")
    filas = evaluar_sondas(sondas, d, sint, args, rng)

    print("\n=== H1: ¿hay atajos? (sondas entrenadas con el 10 % real) ===")
    imprimir_tabla(filas, [ESC_VAL, ESC_TEST])
    print("  Cuanto más lejos de 0.5 con una parte pobre de la imagen, más atajo. Si cae de val a test,\n"
          "  ese atajo no se mantiene en el test desplazado.")

    resumen_h2 = []
    if sint is not None:
        print("\n=== H2: ¿el DDPM amplifica o mitiga el atajo? ===")
        print("Dentro de cada conjunto (validación cruzada): ¿cuánta clase se puede leer de esa parte?")
        imprimir_tabla(filas, [ESC_CV_REAL, ESC_CV_SINT])
        for nombre in sondas:
            r = next(x for x in filas if x["sonda"] == nombre and x["escenario"] == ESC_CV_REAL)
            sr = next(x for x in filas if x["sonda"] == nombre and x["escenario"] == ESC_CV_SINT)
            v = veredicto(r, sr)
            resumen_h2.append({"sonda": nombre, "auc_real": r["auc"], "auc_sintetica": sr["auc"], "veredicto": v})
            print(f"  {nombre:<42s} real {r['auc']:.3f} | sintética {sr['auc']:.3f} -> {v}")
        print("\nTransferencia (¿el atajo de un conjunto sirve en el otro?):")
        imprimir_tabla(filas, [ESC_R_S, ESC_S_VAL, ESC_S_TEST])
        print("  Ojo: las sintéticas son pocas y su IC es más ancho; 'sin diferencia clara' es lo habitual\n"
              "  con pocas imágenes y no significa que no haya diferencia.")

    # ---------- Separación de clases en brillo ----------
    conjuntos = {"real 10 %": (x10, y10), "val": (d["xva"], d["yva"]), "test": (d["xte"], d["yte"])}
    if sint is not None:
        conjuntos["sintéticas"] = sint
    brechas = tabla_brechas(conjuntos, args.k_brillo_borde)
    print("\n=== Separación entre clases (d de Cohen: neumonía - normal, en desviaciones) ===")
    print(f"{'conjunto':<12s}{'medida':<24s}{'normal':>9s}{'neumonía':>10s}{'d':>8s}")
    for b in brechas:
        print(f"{b['conjunto']:<12s}{b['medida']:<24s}{b['media_normal']:>9.3f}{b['media_neumonia']:>10.3f}{b['d_cohen']:>+8.2f}")

    # ---------- Diagnóstico ----------
    diag = []
    if not args.sin_diagnostico:
        diag = diagnostico_falsos_positivos(args.semilla, d, args.k_brillo_borde, elegir_dispositivo())

    # ---------- Guardado ----------
    guardar_csv(filas, DIR_SONDAS / f"sondas_{etiqueta}.csv")
    guardar_csv(brechas, DIR_SONDAS / f"brechas_{etiqueta}.csv")
    with open(DIR_SONDAS / f"sondas_{etiqueta}.json", "w", encoding="utf-8") as f:
        json.dump({"semilla": args.semilla, "sinteticas": args.sinteticas, "sondas": filas, "brechas": brechas,
                   "veredicto_H2": resumen_h2, "diagnostico_falsos_positivos": diag}, f, indent=2, ensure_ascii=False)

    figura_puntos(filas, [ESC_VAL, ESC_TEST], [AZUL, NARANJA],
                  "H1: ¿cuánta clase se lee de cada parte de la imagen real?",
                  DIR_SONDAS / f"h1_sondas_{etiqueta}.png")
    if sint is not None:
        figura_puntos(filas, [ESC_CV_REAL, ESC_CV_SINT], [AZUL, NARANJA],
                      "H2: lo mismo, en imágenes reales y en sintéticas",
                      DIR_SONDAS / f"h2_sondas_{etiqueta}.png")
    print(f"\nResultados guardados en {DIR_SONDAS}")


if __name__ == "__main__":
    main()
