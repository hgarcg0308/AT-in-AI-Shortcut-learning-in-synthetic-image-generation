"""
Escenarios.py
=============
Fases 6, 8 y 9 del flujo de trabajo: selección de sintéticas por incertidumbre,
escenarios experimentales y evaluación con intervalos.

La pregunta de fondo: si al 10 % real le añadimos imágenes sintéticas, ¿el
clasificador generaliza mejor en el test desplazado? Y, ¿da igual qué sintéticas
añadamos?

Escenarios (cada clave se puede activar o no con --escenarios):
    base     10 % real, sin nada más
    clasica  10 % real + copias del propio 10 % real con rotación, escala y brillo
    e1       10 % real + sintéticas elegidas al azar
    e2a      10 % real + sintéticas "fáciles"   (el verificador está muy seguro de su clase)
    e2b      10 % real + sintéticas "medias"    (confianza intermedia)
    e2c      10 % real + sintéticas "frontera"  (el verificador duda)
    e3       10 % real + sintéticas "peores"    (el verificador NO está de acuerdo con la
                                                clase pedida; control negativo)
    e4       solo sintéticas
    techo    todo el train limpio (cota superior con todos los datos reales)

Controles importantes:
  * Mismo número de imágenes: base aparte, todos los escenarios con sintéticas
    añaden la misma cantidad por clase, y 'clasica' añade esa misma cantidad de
    copias transformadas. Así una mejora no puede deberse solo a "más datos".
  * La validación para el early stopping es SIEMPRE la val real limpia.
  * Se evalúa en el test oficial, y por separado en normales y neumonías.
  * Cada escenario se entrena varias veces (--reps) con distinta inicialización.

Ojo: el verificador que elige las sintéticas se entrenó con las mismas ~470 reales,
así que la selección está sesgada hacia lo que ya conoce (limitación del flujo).

Uso:
    python Escenarios.py --semilla 0 --sinteticas outputs/ddpm/sinteticas_semilla0_w2.0.npz

    # Versión rápida para una primera prueba
    python Escenarios.py --semilla 0 --sinteticas outputs/ddpm/sinteticas_semilla0_w2.0.npz \\
        --reps 2 --epocas 30 --bootstrap 300 --escenarios base clasica e1 e2c e3

Salidas en outputs/escenarios/: tablas (.csv, .json), probabilidades (.npz) y figuras (.png).
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

from comun import DIR_SALIDAS, cargar_limpio, cargar_submuestra, elegir_dispositivo, fijar_semilla
from Sondas import AZUL, NARANJA, figura_puntos
from Verificador import CNNVerificadora, a_tensor, cargar_verificador, desplazar, predecir

DIR_ESC = DIR_SALIDAS / "escenarios"

NOMBRES = {
    "base": "Base (10 % real)",
    "clasica": "Aug. clásica (+copias transformadas)",
    "e1": "E1 real + sint. al azar",
    "e2a": "E2a real + sint. fáciles",
    "e2b": "E2b real + sint. medias",
    "e2c": "E2c real + sint. frontera",
    "e3": "E3 real + sint. peores (control)",
    "e4": "E4 solo sintéticas",
    "techo": "Techo (todo el train limpio)",
}
MODO_SELECCION = {"e1": "aleatorio", "e2a": "faciles", "e2b": "medias", "e2c": "frontera", "e3": "peores"}


# ==========================================================================
# 1. Elegir sintéticas según la confianza del verificador (Fase 6)
# ==========================================================================
def confianza_en_clase_pedida(p_neumonia: np.ndarray, y_pedida: np.ndarray) -> np.ndarray:
    """Probabilidad que el verificador da a la clase que se pidió al generar."""
    return np.where(y_pedida == 1, p_neumonia, 1 - p_neumonia)


def seleccionar(conf, y_pedida, k_por_clase: int, modo: str, rng) -> np.ndarray:
    """
    Elige k sintéticas por clase según el modo:
      aleatorio : al azar
      faciles   : las de mayor confianza en la clase pedida
      medias    : las de confianza intermedia (franja central)
      frontera  : las más cercanas a 0.5 (el verificador duda)
      peores    : las de menor confianza en la clase pedida (suelen tener la etiqueta "mal")
    """
    elegidos = []
    for c in (0, 1):
        ic = np.where(y_pedida == c)[0]
        k = min(k_por_clase, len(ic))
        cc = conf[ic]
        if modo == "aleatorio":
            sel = rng.choice(ic, k, replace=False)
        elif modo == "faciles":
            sel = ic[np.argsort(-cc)[:k]]
        elif modo == "peores":
            sel = ic[np.argsort(cc)[:k]]
        elif modo == "frontera":
            sel = ic[np.argsort(np.abs(cc - 0.5))[:k]]
        elif modo == "medias":
            orden = np.argsort(cc)
            ini = (len(ic) - k) // 2
            sel = ic[orden[ini:ini + k]]
        else:
            raise ValueError(modo)
        elegidos.append(sel)
    return np.concatenate(elegidos)


# ==========================================================================
# 2. Aumentación clásica: copias del real con rotación, escala, desplazamiento y brillo
# ==========================================================================
def aumentar_clasica(x: np.ndarray, y: np.ndarray, n: int, rng):
    """Genera n copias transformadas de imágenes reales elegidas al azar."""
    tam = x.shape[1]
    idx = rng.choice(len(x), n, replace=True)
    X = torch.from_numpy(x[idx]).float().div(255).unsqueeze(1)
    ang = torch.from_numpy(rng.uniform(-10, 10, n) * np.pi / 180).float()
    esc = torch.from_numpy(rng.uniform(0.9, 1.1, n)).float()
    tx = torch.from_numpy(rng.uniform(-2, 2, n) * 2 / tam).float()
    ty = torch.from_numpy(rng.uniform(-2, 2, n) * 2 / tam).float()
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = ang.cos() / esc, -ang.sin() / esc, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = ang.sin() / esc, ang.cos() / esc, ty
    grid = F.affine_grid(theta, X.shape, align_corners=False)
    X = F.grid_sample(X, grid, padding_mode="border", align_corners=False)
    # brillo y contraste
    media = X.mean(dim=(1, 2, 3), keepdim=True)
    contraste = torch.from_numpy(rng.uniform(0.9, 1.1, n)).float().view(n, 1, 1, 1)
    brillo = torch.from_numpy(rng.uniform(-0.05, 0.05, n)).float().view(n, 1, 1, 1)
    X = ((X - media) * contraste + media + brillo).clamp(0, 1)
    return (X * 255).round().byte().squeeze(1).numpy(), y[idx]


# ==========================================================================
# 3. Entrenar un clasificador (mismo modelo que el verificador)
# ==========================================================================
def entrenar_clasificador(x, y, xva, yva, dev, semilla, args):
    """Entrena con early stopping sobre la AUC de la val real. Devuelve el modelo ya en su mejor época."""
    fijar_semilla(semilla)
    X, Y = a_tensor(x, dev), torch.from_numpy(y).float().to(dev)
    modelo = CNNVerificadora(args.canales, 0.3).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-3)
    pos_weight = torch.tensor((y == 0).sum() / max(1, (y == 1).sum()), device=dev).float()
    criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    mejor_auc, mejor_loss, mejor_estado, sin_mejora = -1.0, float("inf"), None, 0
    for _ in range(args.epocas):
        modelo.train()
        orden = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), args.batch):
            idx = orden[i:i + args.batch]
            if len(idx) < 2:
                continue
            xb = desplazar(X[idx]) if args.aug_entrenamiento else X[idx]
            loss = criterio(modelo(xb), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        p, _ = predecir(modelo, xva, dev)
        auc = roc_auc_score(yva, p)
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        loss_va = float(-(yva * np.log(pc) + (1 - yva) * np.log(1 - pc)).mean())
        # mejor = más AUC; si empatan, gana la menor pérdida
        if auc > mejor_auc + 1e-4 or (abs(auc - mejor_auc) <= 1e-4 and loss_va < mejor_loss):
            mejor_auc, mejor_loss, sin_mejora = max(auc, mejor_auc), loss_va, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
        if sin_mejora >= args.paciencia:
            break
    modelo.load_state_dict(mejor_estado)
    return modelo


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


def metricas_punto(y, P) -> dict:
    """P: (reps, n) probabilidades. Devuelve la media sobre las réplicas de entrenamiento."""
    aucs = [roc_auc_score(y, p) for p in P]
    pred = P >= 0.5
    return {"auc": float(np.mean(aucs)), "auc_sd_reps": float(np.std(aucs)),
            "sens": float(pred[:, y == 1].mean()), "esp": float((~pred[:, y == 0]).mean()),
            "ece": float(np.mean([ece(y, p) for p in P]))}


def bootstrap_pareado(y, P_por_esc: dict, B: int, rng) -> dict:
    """
    Remuestrea las imágenes del test y calcula las métricas de TODOS los escenarios con
    los mismos remuestreos. Así las diferencias entre escenarios son pareadas (misma
    muestra de test), que es más justo que comparar dos intervalos sueltos.
    """
    n, nombres = len(y), list(P_por_esc)
    out = {m: {s: np.full(B, np.nan) for s in nombres} for m in ("auc", "sens", "esp")}
    for b in range(B):
        i = rng.integers(0, n, n)
        yb = y[i]
        if len(np.unique(yb)) < 2:
            continue
        pos = yb == 1
        for s in nombres:
            pb = P_por_esc[s][:, i]
            pred = pb >= 0.5
            out["auc"][s][b] = np.mean([roc_auc_score(yb, p) for p in pb])
            out["sens"][s][b] = pred[:, pos].mean()
            out["esp"][s][b] = (~pred[:, ~pos]).mean()
    return out


def ic(v) -> tuple:
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return float(lo), float(hi)


# ==========================================================================
# 5. Montar los conjuntos de entrenamiento de cada escenario
# ==========================================================================
def montar_escenarios(args, x10, y10, datos, pool, rng):
    """Devuelve {clave: (x, y)} con el conjunto de entrenamiento de cada escenario."""
    n_real = len(x10)
    xs, ys, conf = pool if pool is not None else (None, None, None)

    k = 0
    if xs is not None:
        disponibles = min((ys == 0).sum(), (ys == 1).sum())
        pedido = args.n_sint_por_clase or int(round(args.ratio * n_real / 2))
        # por defecto no usamos más de la mitad del pool por clase, para que filtrar tenga sentido
        k = min(pedido, disponibles // 2) if args.n_sint_por_clase is None else min(pedido, disponibles)
        k = max(k, 1)
        print(f"Sintéticas disponibles: {disponibles} por clase | se añaden {k} por clase ({2 * k} en total) "
              f"a los {n_real} reales")
        if k >= disponibles:
            print("AVISO: se usa todo el pool, así que los filtros (fáciles/medias/frontera/peores) "
                  "elegirán lo mismo. Genera más sintéticas o baja --n-sint-por-clase.")

    esc = {}
    for clave in args.escenarios:
        if clave == "base":
            esc[clave] = (x10, y10)
        elif clave == "clasica":
            xa, ya = aumentar_clasica(x10, y10, 2 * k, rng)
            esc[clave] = (np.concatenate([x10, xa]), np.concatenate([y10, ya]))
        elif clave in MODO_SELECCION:
            sel = seleccionar(conf, ys, k, MODO_SELECCION[clave], rng)
            esc[clave] = (np.concatenate([x10, xs[sel]]), np.concatenate([y10, ys[sel]]))
        elif clave == "e4":
            kt = max(1, int(round((n_real + 2 * k) / 2)))
            sel = seleccionar(conf if conf is not None else np.zeros(len(ys)), ys, kt, "aleatorio", rng)
            esc[clave] = (xs[sel], ys[sel])
        elif clave == "techo":
            esc[clave] = (datos["train_x"], datos["train_y"])
    return esc


# ==========================================================================
# 6. Salidas
# ==========================================================================
def guardar_csv(filas, ruta):
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        columnas = list(dict.fromkeys(k for fila in filas for k in fila))  # la base no tiene columnas "delta"
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)


def main():
    ap = argparse.ArgumentParser(description="Escenarios experimentales con sintéticas, evaluación con intervalos")
    ap.add_argument("--semilla", type=int, default=0, help="submuestra del 10 %% (y verificador) a usar")
    ap.add_argument("--sinteticas", default=None, help="archivo .npz generado por DDPM.py")
    ap.add_argument("--escenarios", nargs="+", default=list(NOMBRES), choices=list(NOMBRES))
    ap.add_argument("--ratio", type=float, default=1.0, help="sintéticas añadidas = ratio x nº de reales (ej. 1, 2, 5)")
    ap.add_argument("--n-sint-por-clase", type=int, default=None, help="fija cuántas sintéticas por clase se añaden")
    ap.add_argument("--reps", type=int, default=3, help="entrenamientos por escenario (distinta inicialización)")
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--paciencia", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--canales", type=int, default=32)
    ap.add_argument("--aug-entrenamiento", action="store_true",
                    help="añade pequeños desplazamientos al entrenar (por defecto NO, para no mezclarlo con 'clasica')")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    necesita_sint = [e for e in args.escenarios if e not in ("base", "clasica", "techo")]
    if necesita_sint and not args.sinteticas:
        raise SystemExit(f"Los escenarios {necesita_sint} necesitan --sinteticas (archivo de DDPM.py)")

    DIR_ESC.mkdir(parents=True, exist_ok=True)
    dev = elegir_dispositivo()
    rng = np.random.default_rng(args.semilla)

    x10, y10, _, _ = cargar_submuestra(args.semilla)
    datos = cargar_limpio()
    xva, yva, xte, yte = datos["val_x"], datos["val_y"], datos["test_x"], datos["test_y"]
    print(f"Dispositivo: {dev} | real 10 %: {len(x10)} | val limpia: {len(xva)} | test: {len(xte)}")

    # ---------- sintéticas y confianza del verificador ----------
    pool, etiqueta = None, f"semilla{args.semilla}"
    if args.sinteticas:
        s = np.load(args.sinteticas)
        xs, ys = s["images"], s["labels"].astype(int)
        conf = None
        if any(e in MODO_SELECCION and e != "e1" for e in args.escenarios):
            ver = cargar_verificador(args.semilla, dev)  # avisa si no está entrenado
            p_s, _ = predecir(ver, xs, dev)
            conf = confianza_en_clase_pedida(p_s, ys)
            print(f"Verificador: confianza media en la clase pedida = {conf.mean():.3f} "
                  f"| desacuerdo con la clase pedida = {(conf < 0.5).mean() * 100:.1f} %")
        pool = (xs, ys, conf)
        etiqueta = Path(args.sinteticas).stem.replace("sinteticas_", "") + f"_x{args.ratio:g}"

    esc = montar_escenarios(args, x10, y10, datos, pool, rng)

    print("\nConjuntos de entrenamiento:")
    for c, (x, y) in esc.items():
        print(f"  {NOMBRES[c]:<40s} {len(x):5d} imágenes (normal {(y == 0).sum()}, neumonía {(y == 1).sum()})")

    # ---------- entrenar ----------
    P_test, P_val = {}, {}
    t0 = time.time()
    for c, (x, y) in esc.items():
        pt, pv = [], []
        for r in range(args.reps):
            m = entrenar_clasificador(x, y, xva, yva, dev, args.semilla * 100 + r, args)
            pt.append(predecir(m, xte, dev)[0])
            pv.append(predecir(m, xva, dev)[0])
        P_test[NOMBRES[c]], P_val[NOMBRES[c]] = np.array(pt), np.array(pv)
        mp = metricas_punto(yte, P_test[NOMBRES[c]])
        print(f"  [{time.time() - t0:5.0f}s] {NOMBRES[c]:<40s} AUC test {mp['auc']:.3f} "
              f"(± {mp['auc_sd_reps']:.3f} entre réplicas)")

    # ---------- métricas e intervalos ----------
    print(f"\nBootstrap pareado sobre el test ({args.bootstrap} remuestreos)...")
    boot = bootstrap_pareado(yte, P_test, args.bootstrap, rng)
    base = NOMBRES["base"] if "base" in args.escenarios else None

    filas = []
    for nombre in P_test:
        mp = metricas_punto(yte, P_test[nombre])
        mv = metricas_punto(yva, P_val[nombre])
        fila = {"escenario": nombre, "n_entrenamiento": int(len(esc[[k for k, v in NOMBRES.items() if v == nombre][0]][0])),
                "auc_test": round(mp["auc"], 4), "auc_test_sd_reps": round(mp["auc_sd_reps"], 4),
                "auc_test_ic_bajo": round(ic(boot["auc"][nombre])[0], 4), "auc_test_ic_alto": round(ic(boot["auc"][nombre])[1], 4),
                "sens_test": round(mp["sens"], 4), "sens_ic_bajo": round(ic(boot["sens"][nombre])[0], 4),
                "sens_ic_alto": round(ic(boot["sens"][nombre])[1], 4),
                "esp_test": round(mp["esp"], 4), "esp_ic_bajo": round(ic(boot["esp"][nombre])[0], 4),
                "esp_ic_alto": round(ic(boot["esp"][nombre])[1], 4),
                "ece_test": round(mp["ece"], 4), "auc_val": round(mv["auc"], 4)}
        if base and nombre != base:
            for m in ("auc", "esp"):
                d = boot[m][nombre] - boot[m][base]
                lo, hi = ic(d)
                fila[f"delta_{m}_vs_base"] = round(float(np.nanmean(d)), 4)
                fila[f"delta_{m}_ic_bajo"], fila[f"delta_{m}_ic_alto"] = round(lo, 4), round(hi, 4)
        filas.append(fila)

    # ---------- tabla en pantalla ----------
    print("\n=== Resultados en el test oficial (media de las réplicas; IC 95 % por bootstrap pareado) ===")
    print(f"{'escenario':<40s}{'AUC test [IC]':>26s}{'Δ AUC vs base [IC]':>30s}{'sens':>7s}{'esp':>7s}{'ECE':>7s}{'AUC val':>9s}")
    for f in filas:
        d = ""
        if "delta_auc_vs_base" in f:
            marca = "*" if (f["delta_auc_ic_bajo"] > 0 or f["delta_auc_ic_alto"] < 0) else " "
            d = f"{f['delta_auc_vs_base']:+.3f} [{f['delta_auc_ic_bajo']:+.3f}, {f['delta_auc_ic_alto']:+.3f}]{marca}"
        print(f"{f['escenario']:<40s}{f['auc_test']:>9.3f} [{f['auc_test_ic_bajo']:.3f}, {f['auc_test_ic_alto']:.3f}]"
              f"{d:>30s}{f['sens_test']:>7.3f}{f['esp_test']:>7.3f}{f['ece_test']:>7.3f}{f['auc_val']:>9.3f}")
    print("  * = el intervalo de la diferencia no incluye 0.")
    if base:
        print("\n=== Especificidad en test frente a la base (donde falla el desplazamiento) ===")
        for f in filas:
            if "delta_esp_vs_base" in f:
                marca = "*" if (f["delta_esp_ic_bajo"] > 0 or f["delta_esp_ic_alto"] < 0) else " "
                print(f"  {f['escenario']:<40s} esp {f['esp_test']:.3f} | Δ {f['delta_esp_vs_base']:+.3f} "
                      f"[{f['delta_esp_ic_bajo']:+.3f}, {f['delta_esp_ic_alto']:+.3f}]{marca}")
    print("\nCuidado al interpretar: estos intervalos solo recogen el test finito (624 imágenes) y la\n"
          "inicialización; son de UNA submuestra del 10 %. Para decir algo general hay que repetir con\n"
          "otras semillas (--semilla 1..4) y comparar las diferencias entre ellas. Diferencias de AUC de\n"
          "0.01-0.02 normalmente no se distinguen del ruido.")

    # ---------- guardado ----------
    guardar_csv(filas, DIR_ESC / f"escenarios_{etiqueta}.csv")
    with open(DIR_ESC / f"escenarios_{etiqueta}.json", "w", encoding="utf-8") as f:
        json.dump({"semilla": args.semilla, "sinteticas": args.sinteticas, "ratio": args.ratio,
                   "reps": args.reps, "bootstrap": args.bootstrap, "resultados": filas}, f, indent=2, ensure_ascii=False)
    np.savez_compressed(DIR_ESC / f"probabilidades_{etiqueta}.npz", y_test=yte, y_val=yva,
                        **{f"test::{k}": v for k, v in P_test.items()}, **{f"val::{k}": v for k, v in P_val.items()})

    rows_auc = [{"sonda": f["escenario"], "escenario": "AUC en test", "auc": f["auc_test"],
                 "ic_bajo": f["auc_test_ic_bajo"], "ic_alto": f["auc_test_ic_alto"]} for f in filas]
    figura_puntos(rows_auc, ["AUC en test"], [AZUL], "AUC en el test oficial según el escenario",
                  DIR_ESC / f"auc_{etiqueta}.png")
    rows_se = []
    for f in filas:
        rows_se.append({"sonda": f["escenario"], "escenario": "Sensibilidad (neumonías acertadas)", "auc": f["sens_test"],
                        "ic_bajo": f["sens_ic_bajo"], "ic_alto": f["sens_ic_alto"]})
        rows_se.append({"sonda": f["escenario"], "escenario": "Especificidad (normales acertadas)", "auc": f["esp_test"],
                        "ic_bajo": f["esp_ic_bajo"], "ic_alto": f["esp_ic_alto"]})
    figura_puntos(rows_se, ["Sensibilidad (neumonías acertadas)", "Especificidad (normales acertadas)"], [AZUL, NARANJA],
                  "Aciertos por clase en el test (umbral 0.5)", DIR_ESC / f"sens_esp_{etiqueta}.png",
                  xlim=(0, 1.02), xlabel="Proporción acertada (con intervalo del 95 %)", azar=False)
    print(f"\nResultados guardados en {DIR_ESC} (…{etiqueta}.*)")


if __name__ == "__main__":
    main()
