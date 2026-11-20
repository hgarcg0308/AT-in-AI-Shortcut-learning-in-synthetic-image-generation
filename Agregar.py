"""
Agregar.py
==========
Fase 9 completa: junta los resultados de todas las semillas (submuestras del 10 %) y
de todos los valores de guidance, y los convierte en tablas, figuras e informe.

Por qué hace falta: cada ejecución de Escenarios.py usa UNA submuestra del 10 %, y
sus intervalos solo recogen el test finito y la inicialización. Aquí se mira lo que
importa: ¿la diferencia frente a la base se repite semilla tras semilla?

Qué calcula:
  * Escenarios: para cada escenario y cada semilla, la diferencia pareada frente a la
    base (misma submuestra, mismo test). Luego la media entre semillas, su desviación,
    un intervalo del 95 % (t de Student con n-1 grados de libertad) y en cuántas
    semillas la diferencia es positiva.
  * Calidad de las sintéticas por guidance: ruido de etiquetas, memorización y KID.
  * Atajos: sondas sobre real (H1) y comparación real/sintético (H2), con cuántas
    semillas dan "amplifica", "mitiga" o "sin diferencia clara".
  * Verificador base y diagnóstico de falsos positivos frente al brillo del borde.

Uso:
    python Agregar.py

Lee de outputs/escenarios, outputs/verificador y outputs/sondas, y escribe en
outputs/resumen/: resumen.md, varios .csv/.json y las figuras (.png).
"""

import os
# En Windows + conda, NumPy y PyTorch a veces cargan cada uno su copia de OpenMP y
# Python se cierra con "OMP: Error #15". Esta línea lo evita. Debe ir ANTES de importar nada más.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import csv
import json
import re
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from comun import DIR_SALIDAS
from Escenarios import NOMBRES
from Sondas import AZUL, FONDO, REJILLA, TINTA, TINTA_SUAVE

DIR_RES = DIR_SALIDAS / "resumen"
DIR_ESC = DIR_SALIDAS / "escenarios"
DIR_VERIF = DIR_SALIDAS / "verificador"
DIR_SONDAS = DIR_SALIDAS / "sondas"

ORDEN = list(NOMBRES.values())
BASE, TECHO = NOMBRES["base"], NOMBRES["techo"]


# ==========================================================================
# Estadística entre semillas
# ==========================================================================
def resumir(valores) -> dict:
    """Media, desviación e intervalo del 95 % (t de Student) de los valores de cada semilla."""
    v = np.array([x for x in valores if x is not None and np.isfinite(x)], dtype=float)
    n = len(v)
    out = {"n": n, "media": np.nan, "sd": np.nan, "ic_bajo": np.nan, "ic_alto": np.nan, "positivas": 0}
    if n == 0:
        return out
    out["media"] = float(v.mean())
    out["positivas"] = int((v > 0).sum())
    if n >= 2:
        out["sd"] = float(v.std(ddof=1))
        h = stats.t.ppf(0.975, n - 1) * out["sd"] / np.sqrt(n)
        out["ic_bajo"], out["ic_alto"] = out["media"] - h, out["media"] + h
    return out


def excluye_cero(r: dict) -> bool:
    return np.isfinite(r["ic_bajo"]) and (r["ic_bajo"] > 0 or r["ic_alto"] < 0)


def fmt(x, dec=3, signo=False) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:+.{dec}f}" if signo else f"{x:.{dec}f}"


def fmt_ms(r: dict, dec=3) -> str:
    """'media ± sd' con el número de semillas."""
    if r["n"] == 0:
        return "-"
    return f"{fmt(r['media'], dec)} ± {fmt(r['sd'], dec)}" if r["n"] > 1 else f"{fmt(r['media'], dec)} (1 semilla)"


def leer_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def guardar_csv(filas, ruta):
    if not filas:
        return
    columnas = list(dict.fromkeys(k for f in filas for k in f))
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)


def tabla_md(cabecera, filas) -> str:
    lineas = ["| " + " | ".join(cabecera) + " |", "|" + "|".join(["---"] * len(cabecera)) + "|"]
    lineas += ["| " + " | ".join(str(c) for c in f) + " |" for f in filas]
    return "\n".join(lineas)


# ==========================================================================
# 1. Escenarios
# ==========================================================================
RE_ESC = re.compile(r"escenarios_semilla(\d+)_w([\d.]+)_x([\d.]+)\.json$")


def cargar_escenarios():
    """{(w, ratio): {semilla: {escenario: fila}}} y el techo de cada semilla."""
    datos, techo = defaultdict(dict), {}
    for f in sorted(DIR_ESC.glob("escenarios_semilla*.json")):
        m = RE_ESC.search(f.name)
        if not m:
            continue
        s, w, r = int(m[1]), float(m[2]), float(m[3])
        filas = {x["escenario"]: x for x in leer_json(f)["resultados"]}
        datos[(w, r)][s] = filas
        if TECHO in filas:
            techo[s] = filas[TECHO]
    return datos, techo


def agregar_escenarios(datos, techo):
    """Una fila por (guidance, ratio, escenario) con las estadísticas entre semillas."""
    filas, valores = [], {}
    for (w, r), por_semilla in sorted(datos.items()):
        semillas = sorted(por_semilla)
        presentes = {e for s in semillas for e in por_semilla[s]} | ({TECHO} if techo else set())
        for e in [x for x in ORDEN if x in presentes]:
            v = defaultdict(list)
            for s in semillas:
                fila = por_semilla[s].get(e) or (techo.get(s) if e == TECHO else None)
                if fila is None:
                    continue
                v["auc"].append(fila["auc_test"])
                v["sens"].append(fila["sens_test"])
                v["esp"].append(fila["esp_test"])
                v["ece"].append(fila["ece_test"])
                v["d_auc"].append(fila.get("delta_auc_vs_base", 0.0))
                v["d_esp"].append(fila.get("delta_esp_vs_base", 0.0))
            res = {k: resumir(x) for k, x in v.items()}
            valores[(w, r, e)] = (dict(v), res)
            filas.append({"guidance": w, "ratio": r, "escenario": e, "n_semillas": res["auc"]["n"],
                          "auc_media": res["auc"]["media"], "auc_sd": res["auc"]["sd"],
                          "sens_media": res["sens"]["media"], "esp_media": res["esp"]["media"],
                          "esp_sd": res["esp"]["sd"], "ece_media": res["ece"]["media"],
                          "d_auc_media": res["d_auc"]["media"], "d_auc_ic_bajo": res["d_auc"]["ic_bajo"],
                          "d_auc_ic_alto": res["d_auc"]["ic_alto"], "d_auc_semillas_positivas": res["d_auc"]["positivas"],
                          "d_esp_media": res["d_esp"]["media"], "d_esp_ic_bajo": res["d_esp"]["ic_bajo"],
                          "d_esp_ic_alto": res["d_esp"]["ic_alto"], "d_esp_semillas_positivas": res["d_esp"]["positivas"]})
    return filas, valores


def figura_deltas(items, titulo, xlabel, ruta):
    """items: lista de (escenario, valores_por_semilla, resumen). Puntos grises = semillas; azul = media ± IC."""
    n = len(items)
    fig, ax = plt.subplots(figsize=(8.4, 0.75 * n + 2.0), facecolor=FONDO)
    ax.set_facecolor(FONDO)
    todos = []
    for i, (nombre, vals, r) in enumerate(items):
        ax.plot(vals, [i] * len(vals), "o", color=TINTA_SUAVE, alpha=0.55, markersize=5, markeredgewidth=0,
                label="Cada semilla" if i == 0 else None, zorder=2)
        xerr = [[r["media"] - r["ic_bajo"]], [r["ic_alto"] - r["media"]]] if np.isfinite(r["ic_bajo"]) else None
        ax.errorbar([r["media"]], [i], xerr=xerr, fmt="o", color=AZUL, ecolor=AZUL, elinewidth=2, markersize=8,
                    markeredgecolor=FONDO, markeredgewidth=2, label="Media ± IC 95 % entre semillas" if i == 0 else None,
                    zorder=3)
        todos += list(vals) + ([r["ic_bajo"], r["ic_alto"]] if np.isfinite(r["ic_bajo"]) else [r["media"]])
    ax.axvline(0, color=TINTA_SUAVE, linestyle="--", linewidth=1)
    ax.text(0, -0.78, "sin cambio", color=TINTA_SUAVE, fontsize=8, ha="center")
    lim = max(0.02, max(abs(x) for x in todos) * 1.15)
    ax.set_xlim(-lim, lim)
    ax.set_yticks(range(n))
    ax.set_yticklabels([x[0] for x in items], color=TINTA, fontsize=9)
    ax.set_ylim(n - 0.4, -0.9)
    ax.set_xlabel(xlabel, color=TINTA_SUAVE, fontsize=9)
    ax.set_title(titulo, color=TINTA, fontsize=11, loc="left", pad=30)
    ax.grid(axis="x", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(REJILLA)
    ax.tick_params(colors=TINTA_SUAVE, length=0)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              labelcolor=TINTA_SUAVE, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=140, facecolor=FONDO)
    plt.close(fig)


# ==========================================================================
# 2. Calidad de las sintéticas (Verificador.py evaluar)
# ==========================================================================
RE_EVAL = re.compile(r"evaluacion_sinteticas_semilla(\d+)_w([\d.]+)\.json$")


def agregar_calidad():
    por_w = defaultdict(lambda: defaultdict(list))
    for f in sorted(DIR_VERIF.glob("evaluacion_sinteticas_semilla*.json")):
        m = RE_EVAL.search(f.name)
        if not m:
            continue
        w, d = float(m[2]), leer_json(f)
        rn, me, ki = d["ruido_etiquetas"], d["memorizacion"], d.get("kid", {})
        g = por_w[w]
        g["desacuerdo_normal"].append(rn["desacuerdo_sinteticas"]["normal"])
        g["desacuerdo_neumonia"].append(rn["desacuerdo_sinteticas"]["neumonia"])
        g["error_val_normal"].append(rn["error_en_val_real_(referencia)"]["normal"])
        g["error_val_neumonia"].append(rn["error_en_val_real_(referencia)"]["neumonia"])
        g["r_mediana_sint"].append(me["r_max_mediana_sinteticas"])
        g["r_mediana_val"].append(me["r_max_mediana_val_real"])
        g["frac_copias"].append(me["frac_sinteticas_r_mayor_0.99"])
        g["frac_sobre_p95"].append(me["frac_sinteticas_por_encima_percentil95_val"])
        for c in ("normal", "neumonia"):
            if c in ki:
                g[f"kid_{c}"].append(ki[c]["sinteticas_vs_train10"][0])
                g[f"kid_ref_{c}"].append(ki[c]["val_real_vs_train10_(referencia)"][0])
    filas = []
    for w in sorted(por_w):
        fila = {"guidance": w}
        for k, vals in por_w[w].items():
            r = resumir(vals)
            fila[k], fila[k + "_sd"], fila["n_semillas"] = r["media"], r["sd"], r["n"]
        filas.append(fila)
    return filas


# ==========================================================================
# 3. Sondas de atajo (Sondas.py)
# ==========================================================================
RE_S1 = re.compile(r"sondas_semilla(\d+)\.json$")
RE_S2 = re.compile(r"sondas_semilla(\d+)_w([\d.]+)\.json$")
ESC_VAL, ESC_TEST = "real 10 % → val", "real 10 % → test"
ESC_CV_REAL, ESC_CV_SINT = "real 10 % (validación cruzada)", "sintéticas (validación cruzada)"


def agregar_sondas():
    h1, h2, brechas, diag, veredictos = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), \
        defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    vistos_h1 = set()
    for f in sorted(DIR_SONDAS.glob("sondas_semilla*.json")):
        m1, m2 = RE_S1.search(f.name), RE_S2.search(f.name)
        if not (m1 or m2):
            continue
        s = int((m1 or m2)[1])
        d = leer_json(f)
        # H1 y diagnóstico: una sola vez por semilla (sale igual en el archivo sin o con sintéticas)
        if s not in vistos_h1:
            vistos_h1.add(s)
            for x in d["sondas"]:
                if x["escenario"] in (ESC_VAL, ESC_TEST):
                    h1[x["sonda"]][x["escenario"]].append(x["auc"])
            for x in d.get("diagnostico_falsos_positivos", []):
                diag[x["conjunto"]].append((x["pct_falsos_positivos"], x["rho_brillo_borde"]))
        if m2:
            w = float(m2[2])
            for x in d["sondas"]:
                if x["escenario"] in (ESC_CV_REAL, ESC_CV_SINT):
                    h2[(w, x["sonda"])][x["escenario"]].append(x["auc"])
            for x in d.get("veredicto_H2", []):
                v = x["veredicto"]
                clave = "amplifica" if "MÁS" in v else ("mitiga" if "MENOS" in v else "sin_diferencia")
                veredictos[w][x["sonda"]][clave] += 1
            for b in d.get("brechas", []):
                if b["medida"].startswith("brillo"):
                    brechas[(w, b["conjunto"], b["medida"])].append(b["d_cohen"])
    return h1, h2, brechas, diag, veredictos


# ==========================================================================
# 4. Verificador base
# ==========================================================================
def agregar_verificador():
    v = defaultdict(lambda: defaultdict(list))
    for f in sorted(DIR_VERIF.glob("verificador_semilla*_metricas.json")):
        d = leer_json(f)
        for conj in ("val", "test"):
            for k in ("auc", "sensibilidad", "especificidad", "accuracy"):
                v[conj][k].append(d[conj][k])
    return v


# ==========================================================================
# Programa principal
# ==========================================================================
def main():
    DIR_RES.mkdir(parents=True, exist_ok=True)
    md = [f"# Resumen del flujo entre semillas\n\nGenerado el {datetime.now():%d/%m/%Y %H:%M} por `Agregar.py`.\n"]

    # ---------- verificador base ----------
    ver = agregar_verificador()
    if ver:
        n = len(ver["val"]["auc"])
        md.append(f"## 1. Clasificador base (verificador), {n} semilla(s)\n")
        filas = [[conj] + [fmt_ms(resumir(ver[conj][k])) for k in ("auc", "sensibilidad", "especificidad", "accuracy")]
                 for conj in ("val", "test")]
        md.append(tabla_md(["conjunto", "AUC", "sensibilidad", "especificidad", "exactitud"], filas))
        md.append("\nLa especificidad en test cae respecto a val aunque el AUC se mantenga: aparece el desplazamiento.\n")
        print(f"[verificador base] {n} semilla(s): AUC test {fmt_ms(resumir(ver['test']['auc']))} | "
              f"especificidad test {fmt_ms(resumir(ver['test']['especificidad']))}")

    # ---------- calidad de sintéticas ----------
    calidad = agregar_calidad()
    if calidad:
        guardar_csv(calidad, DIR_RES / "calidad_sinteticas.csv")
        md.append("## 2. Calidad de las sintéticas según el guidance\n")
        filas = []
        for c in calidad:
            filas.append([f"{c['guidance']:g}", c.get("n_semillas", "-"),
                          f"{fmt(c['desacuerdo_normal'] * 100, 1)} % / {fmt(c['desacuerdo_neumonia'] * 100, 1)} %",
                          f"{fmt(c['error_val_normal'] * 100, 1)} % / {fmt(c['error_val_neumonia'] * 100, 1)} %",
                          f"{fmt(c['r_mediana_sint'])} ({fmt(c['r_mediana_val'])})",
                          f"{fmt(c['frac_sobre_p95'] * 100, 1)} %", f"{fmt(c['frac_copias'] * 100, 1)} %",
                          fmt(c.get("kid_normal"), 2), fmt(c.get("kid_neumonia"), 2)])
        md.append(tabla_md(["guidance", "semillas", "desacuerdo etiqueta (normal / neumonía)",
                            "error en val real (ref.)", "r mediana sint. (val real)", "sint. > p95 de val",
                            "casi copias r>0.99", "KID normal", "KID neumonía"], filas))
        md.append("\n*Desacuerdo bajo = sintéticas fáciles de clasificar. 'sint. > p95 de val' esperado ≈ 5 %. "
                  "El KID tiene escala propia: solo sirve para comparar entre guidance.*\n")

    # ---------- sondas ----------
    h1, h2, brechas, diag, veredictos = agregar_sondas()
    if h1:
        md.append("## 3. Atajos en las imágenes reales (H1): AUC de cada sonda\n")
        filas = [[sonda, fmt_ms(resumir(esc[ESC_VAL])), fmt_ms(resumir(esc[ESC_TEST]))] for sonda, esc in h1.items()]
        md.append(tabla_md(["sonda", "AUC → val", "AUC → test"], filas))
        md.append("\nSi una parte pobre de la imagen (brillo, borde) ya da un AUC alto, esa parte contiene señal de clase.\n")
        guardar_csv([{"sonda": s, "auc_val": resumir(e[ESC_VAL])["media"], "auc_test": resumir(e[ESC_TEST])["media"]}
                     for s, e in h1.items()], DIR_RES / "sondas_h1.csv")
    if diag:
        md.append("### Diagnóstico: falsos positivos en normales y brillo del borde\n")
        filas = [[conj, fmt_ms(resumir([x[0] for x in v]), 1), fmt_ms(resumir([x[1] for x in v]), 2)] for conj, v in diag.items()]
        md.append(tabla_md(["conjunto", "% falsos positivos", "rho(P, brillo del borde)"], filas))
        md.append("")
    if h2:
        md.append("## 4. ¿El DDPM amplifica o mitiga el atajo? (H2)\n")
        filas, csv_h2 = [], []
        for (w, sonda), esc in sorted(h2.items()):
            r, sn = resumir(esc[ESC_CV_REAL]), resumir(esc[ESC_CV_SINT])
            v = veredictos[w][sonda]
            filas.append([f"{w:g}", sonda, fmt_ms(r), fmt_ms(sn),
                          f"{v['amplifica']} / {v['mitiga']} / {v['sin_diferencia']}"])
            csv_h2.append({"guidance": w, "sonda": sonda, "auc_real": r["media"], "auc_sintetica": sn["media"],
                           "semillas_amplifica": v["amplifica"], "semillas_mitiga": v["mitiga"],
                           "semillas_sin_diferencia": v["sin_diferencia"]})
        md.append(tabla_md(["guidance", "sonda", "AUC real (CV)", "AUC sintética (CV)",
                            "semillas: amplifica / mitiga / sin diferencia"], filas))
        guardar_csv(csv_h2, DIR_RES / "sondas_h2.csv")
        md.append("\nCada semilla decide con los intervalos de su propia validación cruzada; con pocas sintéticas "
                  "'sin diferencia' es lo habitual y no prueba que no haya diferencia.\n")
    if brechas:
        md.append("### Separación entre clases en brillo (d de Cohen, neumonía − normal)\n")
        filas = [[f"{w:g}", conj, medida, fmt_ms(resumir(v), 2)] for (w, conj, medida), v in sorted(brechas.items())
                 if conj in ("real 10 %", "test", "sintéticas")]
        md.append(tabla_md(["guidance", "conjunto", "medida", "d de Cohen"], filas))
        md.append("\nSi las sintéticas separan las clases en brillo más que el 10 % real, el generador exagera esa diferencia.\n")

    # ---------- escenarios ----------
    datos, techo = cargar_escenarios()
    filas_esc, valores = agregar_escenarios(datos, techo)
    if filas_esc:
        guardar_csv(filas_esc, DIR_RES / "escenarios_resumen.csv")
        with open(DIR_RES / "escenarios_resumen.json", "w", encoding="utf-8") as f:
            json.dump(filas_esc, f, indent=2, ensure_ascii=False, default=float)
        md.append("## 5. Escenarios: diferencia frente a la base, entre semillas\n")
        md.append("Cada diferencia es **pareada** (misma submuestra y mismo test) y se calcula por semilla. "
                  "`*` marca que el intervalo del 95 % entre semillas no incluye 0. "
                  "'Semillas +' es en cuántas semillas la diferencia es positiva.\n")
        for (w, r) in sorted({(k[0], k[1]) for k in valores}):
            md.append(f"### Guidance {w:g}, ratio ×{r:g}\n")
            filas, items_auc, items_esp = [], [], []
            for e in ORDEN:
                if (w, r, e) not in valores:
                    continue
                v, res = valores[(w, r, e)]
                da, de = res["d_auc"], res["d_esp"]
                if e == BASE:
                    filas.append([e, res["auc"]["n"], fmt_ms(res["auc"]), fmt_ms(res["esp"]), "-", "-", "-", "-"])
                    continue
                filas.append([e, res["auc"]["n"], fmt_ms(res["auc"]), fmt_ms(res["esp"]),
                              f"{fmt(da['media'], 3, True)} [{fmt(da['ic_bajo'], 3, True)}, {fmt(da['ic_alto'], 3, True)}]"
                              + (" *" if excluye_cero(da) else ""), f"{da['positivas']}/{da['n']}",
                              f"{fmt(de['media'], 3, True)} [{fmt(de['ic_bajo'], 3, True)}, {fmt(de['ic_alto'], 3, True)}]"
                              + (" *" if excluye_cero(de) else ""), f"{de['positivas']}/{de['n']}"])
                items_auc.append((e, v["d_auc"], da))
                items_esp.append((e, v["d_esp"], de))
            md.append(tabla_md(["escenario", "semillas", "AUC test", "especificidad test", "Δ AUC vs base [IC]",
                                "semillas +", "Δ especificidad vs base [IC]", "semillas +"], filas))
            md.append("")
            if items_auc:
                tag = f"w{w:g}_x{r:g}"
                figura_deltas(items_auc, f"Cambio de AUC frente a la base (guidance {w:g}, ×{r:g})",
                              "Δ AUC en test (media de las réplicas de entrenamiento)", DIR_RES / f"delta_auc_{tag}.png")
                figura_deltas(items_esp, f"Cambio de especificidad frente a la base (guidance {w:g}, ×{r:g})",
                              "Δ especificidad en test (umbral 0.5)", DIR_RES / f"delta_esp_{tag}.png")
            print(f"\n[guidance {w:g}, ratio ×{r:g}]")
            for f in filas:
                print("  " + " | ".join(str(c) for c in f))
        n_max = max(res["auc"]["n"] for _, res in valores.values())
        if n_max < 5:
            md.append(f"> **Ojo:** solo hay {n_max} semilla(s). Con tan pocas, los intervalos entre semillas son muy "
                      "anchos (o no existen con 1). Lanza las 5 submuestras para conclusiones más firmes.\n")

    if not filas_esc and not ver and not calidad and not h1:
        raise SystemExit("No he encontrado resultados en outputs/. Ejecuta antes Pipeline.py (o los scripts sueltos).")

    md.append("## Limitaciones a declarar\n\n"
              "- Cada semilla es una submuestra distinta del 10 % con su propio generador y verificador; "
              "las diferencias se miden dentro de cada semilla y luego se juntan.\n"
              "- El intervalo entre semillas usa una t de Student con pocos grados de libertad: es honesto pero ancho.\n"
              "- Diferencias de AUC de 0.01–0.02 suelen ser indistinguibles del ruido.\n"
              "- El verificador que elige las sintéticas se entrenó con las mismas reales, por lo que la selección "
              "está sesgada hacia lo que ya conoce.\n"
              "- Sin identificador de paciente no se descarta fuga a nivel de paciente; un solo dataset y un solo generador.\n")
    with open(DIR_RES / "resumen.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nResumen guardado en {DIR_RES} (resumen.md, tablas .csv y figuras .png)")


if __name__ == "__main__":
    main()
