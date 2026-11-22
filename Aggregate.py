"""
Aggregate.py
============
Fase 9 completa: junta los resultados de todas las semillas (submuestras del 10 %) y
de todos los valores de guidance, y los convierte en tablas, figuras e informe.

Por qué hace falta: cada ejecución de Scenarios.py usa UNA submuestra del 10 %, y
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
  * Desplazamiento train vs test (lo calcula Shift.py): prevalencia, clasificador de
    dominio, KS y mapa de diferencias.

Uso:
    python Aggregate.py

Lee de outputs/scenarios, outputs/verifier, outputs/probes y outputs/shift, y escribe en
outputs/summary/: summary.md, varios .csv/.json y las figuras (.png).
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

from common import OUTPUT_DIR
from Scenarios import NAMES
from Probes import BLUE, BACKGROUND, GRID, INK, INK_SOFT

SUMMARY_DIR = OUTPUT_DIR / "summary"
SCENARIOS_DIR = OUTPUT_DIR / "scenarios"
VERIFIER_DIR = OUTPUT_DIR / "verifier"
PROBES_DIR = OUTPUT_DIR / "probes"
SHIFT_DIR = OUTPUT_DIR / "shift"

ORDER = list(NAMES.values())
BASE, CEILING = NAMES["base"], NAMES["ceiling"]


# ==========================================================================
# Estadística entre semillas
# ==========================================================================
def summarize(values) -> dict:
    """Media, desviación e intervalo del 95 % (t de Student) de los valores de cada semilla."""
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = len(v)
    out = {"n": n, "mean": np.nan, "sd": np.nan, "ci_low": np.nan, "ci_high": np.nan, "positive": 0}
    if n == 0:
        return out
    out["mean"] = float(v.mean())
    out["positive"] = int((v > 0).sum())
    if n >= 2:
        out["sd"] = float(v.std(ddof=1))
        h = stats.t.ppf(0.975, n - 1) * out["sd"] / np.sqrt(n)
        out["ci_low"], out["ci_high"] = out["mean"] - h, out["mean"] + h
    return out


def excludes_zero(r: dict) -> bool:
    return np.isfinite(r["ci_low"]) and (r["ci_low"] > 0 or r["ci_high"] < 0)


def fmt(x, dec=3, sign=False) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:+.{dec}f}" if sign else f"{x:.{dec}f}"


def fmt_mean_sd(r: dict, dec=3) -> str:
    """'media ± sd' con el número de semillas."""
    if r["n"] == 0:
        return "-"
    return f"{fmt(r['mean'], dec)} ± {fmt(r['sd'], dec)}" if r["n"] > 1 else f"{fmt(r['mean'], dec)} (1 seed)"


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_csv(rows, path):
    if not rows:
        return
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def md_table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# ==========================================================================
# 1. Escenarios
# ==========================================================================
RE_SC = re.compile(r"scenarios_seed(\d+)_w([\d.]+)_x([\d.]+)\.json$")


def load_scenarios():
    """{(w, ratio): {seed: {scenario: row}}} y el techo de cada semilla."""
    data, ceiling = defaultdict(dict), {}
    for f in sorted(SCENARIOS_DIR.glob("scenarios_seed*.json")):
        m = RE_SC.search(f.name)
        if not m:
            continue
        s, w, r = int(m[1]), float(m[2]), float(m[3])
        rows = {x["scenario"]: x for x in read_json(f)["results"]}
        data[(w, r)][s] = rows
        if CEILING in rows:
            # se guarda también la base del mismo archivo, para las diferencias pareadas de sens. y ECE
            ceiling[s] = dict(rows[CEILING], _base=rows.get(BASE))
    return data, ceiling


def paired_diff(row, base_row, key, is_base):
    """row[key] - base_row[key] (0 para la propia base, None si no hay base con la que comparar)."""
    if is_base:
        return 0.0
    if base_row is None or row.get(key) is None or base_row.get(key) is None:
        return None
    return row[key] - base_row[key]


def aggregate_selection():
    """
    Lee de cada JSON de escenarios qué eligió cada filtro (% de imágenes añadidas donde el
    verificador no está de acuerdo con la clase pedida, confianza y epistémica medias) y la
    correlación de Spearman entre las puntuaciones de selección.
    Devuelve ({(w, r): {escenario: {métrica: [valores por semilla]}}}, {(w, par): {semilla: rho}}).
    """
    sel = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    corr = defaultdict(dict)
    for f in sorted(SCENARIOS_DIR.glob("scenarios_seed*.json")):
        m = RE_SC.search(f.name)
        if not m:
            continue
        seed, w, r = int(m[1]), float(m[2]), float(m[3])
        d = read_json(f)
        for name, dg in d.get("selection_diagnostics", {}).items():
            for k, v in dg.items():
                sel[(w, r)][name][k].append(v)
        for pair, rho in d.get("score_spearman", {}).items():
            corr[(w, pair)][seed] = rho  # el pool es el mismo para todos los ratios: una vez por semilla
    return sel, corr


def aggregate_scenarios(data, ceiling):
    """Una fila por (guidance, ratio, escenario) con las estadísticas entre semillas."""
    rows, values = [], {}
    for (w, r), by_seed in sorted(data.items()):
        seeds = sorted(by_seed)
        present = {e for s in seeds for e in by_seed[s]} | ({CEILING} if ceiling else set())
        for e in [x for x in ORDER if x in present]:
            v = defaultdict(list)
            for s in seeds:
                row = by_seed[s].get(e) or (ceiling.get(s) if e == CEILING else None)
                if row is None:
                    continue
                v["auc"].append(row["auc_test"])
                v["sens"].append(row["sens_test"])
                v["spec"].append(row["spec_test"])
                v["ece"].append(row["ece_test"])
                v["d_auc"].append(row.get("delta_auc_vs_base", 0.0))
                v["d_spec"].append(row.get("delta_spec_vs_base", 0.0))
                # sensibilidad y ECE: diferencia pareada de estimaciones puntuales (misma semilla y mismo
                # test) frente a la base; no hay bootstrap, el intervalo es solo entre semillas
                base_row = row.get("_base") or by_seed[s].get(BASE)
                v["d_sens"].append(paired_diff(row, base_row, "sens_test", e == BASE))
                v["d_ece"].append(paired_diff(row, base_row, "ece_test", e == BASE))
            res = {k: summarize(x) for k, x in v.items()}
            values[(w, r, e)] = (dict(v), res)
            rows.append({"guidance": w, "ratio": r, "scenario": e, "n_seeds": res["auc"]["n"],
                         "auc_mean": res["auc"]["mean"], "auc_sd": res["auc"]["sd"],
                         "sens_mean": res["sens"]["mean"], "spec_mean": res["spec"]["mean"],
                         "spec_sd": res["spec"]["sd"], "ece_mean": res["ece"]["mean"],
                         "d_auc_mean": res["d_auc"]["mean"], "d_auc_ci_low": res["d_auc"]["ci_low"],
                         "d_auc_ci_high": res["d_auc"]["ci_high"], "d_auc_positive_seeds": res["d_auc"]["positive"],
                         "d_spec_mean": res["d_spec"]["mean"], "d_spec_ci_low": res["d_spec"]["ci_low"],
                         "d_spec_ci_high": res["d_spec"]["ci_high"], "d_spec_positive_seeds": res["d_spec"]["positive"],
                         "d_sens_mean": res["d_sens"]["mean"], "d_sens_ci_low": res["d_sens"]["ci_low"],
                         "d_sens_ci_high": res["d_sens"]["ci_high"], "d_sens_positive_seeds": res["d_sens"]["positive"],
                         "d_ece_mean": res["d_ece"]["mean"], "d_ece_ci_low": res["d_ece"]["ci_low"],
                         "d_ece_ci_high": res["d_ece"]["ci_high"], "d_ece_positive_seeds": res["d_ece"]["positive"]})
    return rows, values


def delta_figure(items, title, xlabel, path):
    """items: lista de (escenario, valores_por_semilla, resumen). Puntos grises = semillas; azul = media ± IC."""
    n = len(items)
    fig, ax = plt.subplots(figsize=(8.4, 0.75 * n + 2.0), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    everything = []
    for i, (name, vals, r) in enumerate(items):
        ax.plot(vals, [i] * len(vals), "o", color=INK_SOFT, alpha=0.55, markersize=5, markeredgewidth=0,
                label="Each seed" if i == 0 else None, zorder=2)
        xerr = [[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]] if np.isfinite(r["ci_low"]) else None
        ax.errorbar([r["mean"]], [i], xerr=xerr, fmt="o", color=BLUE, ecolor=BLUE, elinewidth=2, markersize=8,
                    markeredgecolor=BACKGROUND, markeredgewidth=2, label="Mean ± 95% CI across seeds" if i == 0 else None,
                    zorder=3)
        everything += list(vals) + ([r["ci_low"], r["ci_high"]] if np.isfinite(r["ci_low"]) else [r["mean"]])
    ax.axvline(0, color=INK_SOFT, linestyle="--", linewidth=1)
    ax.text(0, -0.78, "no change", color=INK_SOFT, fontsize=8, ha="center")
    lim = max(0.02, max(abs(x) for x in everything) * 1.15)
    ax.set_xlim(-lim, lim)
    ax.set_yticks(range(n))
    ax.set_yticklabels([x[0] for x in items], color=INK, fontsize=9)
    ax.set_ylim(n - 0.4, -0.9)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=30)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              labelcolor=INK_SOFT, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=BACKGROUND)
    plt.close(fig)


# ==========================================================================
# 2. Calidad de las sintéticas (Verifier.py evaluate)
# ==========================================================================
RE_EVAL = re.compile(r"evaluation_synthetic_seed(\d+)_w([\d.]+)\.json$")


def aggregate_quality():
    by_w = defaultdict(lambda: defaultdict(list))
    for f in sorted(VERIFIER_DIR.glob("evaluation_synthetic_seed*.json")):
        m = RE_EVAL.search(f.name)
        if not m:
            continue
        w, d = float(m[2]), read_json(f)
        ln, me, ki = d["label_noise"], d["memorization"], d.get("kid", {})
        g = by_w[w]
        g["disagreement_normal"].append(ln["synthetic_disagreement"]["normal"])
        g["disagreement_pneumonia"].append(ln["synthetic_disagreement"]["pneumonia"])
        g["val_error_normal"].append(ln["error_on_real_val_(reference)"]["normal"])
        g["val_error_pneumonia"].append(ln["error_on_real_val_(reference)"]["pneumonia"])
        g["r_median_synth"].append(me["r_max_median_synthetic"])
        g["r_median_val"].append(me["r_max_median_real_val"])
        g["frac_copies"].append(me["frac_synthetic_r_above_0.99"])
        g["frac_above_p95"].append(me["frac_synthetic_above_val_percentile95"])
        for c in ("normal", "pneumonia"):
            if c in ki:
                g[f"kid_{c}"].append(ki[c]["synthetic_vs_train10"][0])
                g[f"kid_ref_{c}"].append(ki[c]["real_val_vs_train10_(reference)"][0])
        for c, pr in d.get("precision_recall", {}).items():  # ausentes en evaluaciones antiguas
            g[f"precision_{c}"].append(pr["synthetic_vs_train10"]["precision"][0])
            g[f"recall_{c}"].append(pr["synthetic_vs_train10"]["recall"][0])
            g[f"precision_ref_{c}"].append(pr["real_val_vs_train10_(reference)"]["precision"][0])
            g[f"recall_ref_{c}"].append(pr["real_val_vs_train10_(reference)"]["recall"][0])
        unc = d.get("uncertainty")
        if unc:
            g["mi_synth"].append(unc["synthetic"]["all"]["mean_mutual_info"])
            g["mi_val"].append(unc["real_val_(reference)"]["all"]["mean_mutual_info"])
    rows = []
    for w in sorted(by_w):
        row = {"guidance": w}
        for k, vals in by_w[w].items():
            r = summarize(vals)
            row[k], row[k + "_sd"] = r["mean"], r["sd"]
        row["n_seeds"] = len(by_w[w]["disagreement_normal"])
        rows.append(row)
    return rows


# ==========================================================================
# 3. Sondas de atajo (Probes.py)
# ==========================================================================
RE_P1 = re.compile(r"probes_seed(\d+)\.json$")
RE_P2 = re.compile(r"probes_seed(\d+)_w([\d.]+)\.json$")
SC_VAL, SC_TEST = "real 10% → val", "real 10% → test"
SC_CV_REAL, SC_CV_SYNTH = "real 10% (cross-validation)", "synthetic (cross-validation)"


def aggregate_probes():
    h1, h2, gaps, diag, verdicts = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), \
        defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    seen_h1 = set()
    for f in sorted(PROBES_DIR.glob("probes_seed*.json")):
        m1, m2 = RE_P1.search(f.name), RE_P2.search(f.name)
        if not (m1 or m2):
            continue
        s = int((m1 or m2)[1])
        d = read_json(f)
        # H1 y diagnóstico: una sola vez por semilla (sale igual en el archivo sin o con sintéticas)
        if s not in seen_h1:
            seen_h1.add(s)
            for x in d["probes"]:
                if x["scenario"] in (SC_VAL, SC_TEST):
                    h1[x["probe"]][x["scenario"]].append(x["auc"])
            for x in d.get("false_positive_diagnosis", []):
                diag[x["set"]].append((x["pct_false_positives"], x["rho_edge_brightness"]))
        if m2:
            w = float(m2[2])
            for x in d["probes"]:
                if x["scenario"] in (SC_CV_REAL, SC_CV_SYNTH):
                    h2[(w, x["probe"])][x["scenario"]].append(x["auc"])
            for x in d.get("verdict_H2", []):
                v = x["verdict"]
                key = "amplifies" if "MORE" in v else ("mitigates" if "LESS" in v else "no_difference")
                verdicts[w][x["probe"]][key] += 1
            for b in d.get("gaps", []):
                if b["measure"].startswith("brightness"):
                    gaps[(w, b["set"], b["measure"])].append(b["cohen_d"])
    return h1, h2, gaps, diag, verdicts


# ==========================================================================
# 4. Verificador base
# ==========================================================================
def aggregate_verifier():
    v = defaultdict(lambda: defaultdict(list))
    for f in sorted(VERIFIER_DIR.glob("verifier_seed*_metrics.json")):
        d = read_json(f)
        for subset in ("val", "test"):
            for k in ("auc", "sensitivity", "specificity", "accuracy"):
                v[subset][k].append(d[subset][k])
    return v


# ==========================================================================
# 5. Desplazamiento train vs test (Shift.py)
# ==========================================================================
def shift_section():
    """Devuelve las líneas markdown de la sección de desplazamiento, o None si no hay shift.json."""
    path = SHIFT_DIR / "shift.json"
    if not path.exists():
        return None
    d = read_json(path)
    md = ["## 0. Dataset shift: clean train vs official test\n"]

    pr = d["prior_shift"]
    rows = [[k, pr[k]["n"], f"{pr[k]['pct_pneumonia']:.1f} % [{pr[k]['ci_low']:.1f}, {pr[k]['ci_high']:.1f}]"]
            for k in ("train", "val", "test")]
    md.append("### Prior (label) shift\n")
    md.append(md_table(["set", "n", "% pneumonia [95 % CI]"], rows))
    dp, w = pr["diff_test_minus_train_pp"], pr["importance_weights_test_over_train"]
    md.append(f"\nTest − train: {dp['value']:+.1f} pp [{dp['ci_low']:+.1f}, {dp['ci_high']:+.1f}]. "
              f"Importance weights p_test(y)/p_train(y): normal {w['normal']:.2f}, pneumonia {w['pneumonia']:.2f}.\n")

    md.append("### Domain classifier: can an image be told apart as train or test?\n")
    rows = []
    for r in d["domain"]:
        ref = "-" if r["auc_train_vs_val_ref"] is None else fmt(r["auc_train_vs_val_ref"])
        rows.append([r["group"], r["region"], f"{r['n_train']} / {r['n_test']}",
                     f"{fmt(r['auc_train_vs_test'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]",
                     f"{fmt(r['auc_null'])} [{fmt(r['null_ci_low'])}, {fmt(r['null_ci_high'])}]", ref,
                     "yes" if r["distinguishable_from_null"] else "no"])
    md.append(md_table(["class", "region", "n train / test", "AUC train vs test [CI]",
                        "null AUC (train split at random) [CI]", "AUC train vs val (ref.)",
                        "above null?"], rows))
    md.append("\nAUC 0.5 = indistinguishable. 'Above null' means the lower end of the interval is above the upper end "
              "of the null interval. A high AUC in one class only is class-conditional shift. 'edge' is the outer ring "
              f"of {d['settings']['k_center']} px and 'center' the rest of the image.\n")

    md.append("### Kolmogorov-Smirnov: distance D between the train and test distributions\n")
    rows = [[r["group"], r["feature"], f"{fmt(r['D'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]",
             fmt(r["noise_floor_p95"]), fmt(r["median_test_minus_train"], 3, True),
             "yes" if r["above_noise_floor"] else "no"] for r in d["ks"]]
    md.append(md_table(["class", "statistic", "D [95 % CI]", "noise floor (p95)", "median test − train",
                        "D above noise?"], rows))
    md.append("\nD is reported instead of a p-value (with large n everything is 'significant'). The bootstrap interval of "
              "D tends to sit slightly above the real value when D is small, so compare D with the noise floor: the 95th "
              "percentile of D if train and test came from the same distribution. KS is univariate and ignores spatial "
              "structure; the domain classifier complements it. The 'all' rows mix the prior shift with the image shift.\n")
    if (SHIFT_DIR / "shift_difference_map.png").exists():
        md.append("![Mean difference test − train, by class and by distance to the border](../shift/shift_difference_map.png)\n")
    return md


# ==========================================================================
# Programa principal
# ==========================================================================
def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    md = [f"# Summary of the pipeline across seeds\n\nGenerated on {datetime.now():%d/%m/%Y %H:%M} by `Aggregate.py`.\n"]

    # ---------- desplazamiento train vs test ----------
    shift_md = shift_section()
    if shift_md:
        md += shift_md
        print(f"[shift] included from {SHIFT_DIR / 'shift.json'}")

    # ---------- verificador base ----------
    ver = aggregate_verifier()
    if ver:
        n = len(ver["val"]["auc"])
        md.append(f"## 1. Base classifier (verifier), {n} seed(s)\n")
        rows = [[subset] + [fmt_mean_sd(summarize(ver[subset][k])) for k in ("auc", "sensitivity", "specificity", "accuracy")]
                for subset in ("val", "test")]
        md.append(md_table(["set", "AUC", "sensitivity", "specificity", "accuracy"], rows))
        md.append("\nTest specificity drops relative to val even though AUC holds: the shift shows up.\n")
        print(f"[base verifier] {n} seed(s): test AUC {fmt_mean_sd(summarize(ver['test']['auc']))} | "
              f"test specificity {fmt_mean_sd(summarize(ver['test']['specificity']))}")

    # ---------- calidad de sintéticas ----------
    quality = aggregate_quality()
    if quality:
        save_csv(quality, SUMMARY_DIR / "synthetic_quality.csv")
        md.append("## 2. Quality of the synthetic images by guidance\n")
        rows = []
        for c in quality:
            rows.append([f"{c['guidance']:g}", c.get("n_seeds", "-"),
                         f"{fmt(c['disagreement_normal'] * 100, 1)} % / {fmt(c['disagreement_pneumonia'] * 100, 1)} %",
                         f"{fmt(c['val_error_normal'] * 100, 1)} % / {fmt(c['val_error_pneumonia'] * 100, 1)} %",
                         f"{fmt(c['r_median_synth'])} ({fmt(c['r_median_val'])})",
                         f"{fmt(c['frac_above_p95'] * 100, 1)} %", f"{fmt(c['frac_copies'] * 100, 1)} %",
                         fmt(c.get("kid_normal"), 2), fmt(c.get("kid_pneumonia"), 2)])
        md.append(md_table(["guidance", "seeds", "label disagreement (normal / pneumonia)",
                            "error on real val (ref.)", "median r synth. (real val)", "synth. > val p95",
                            "near copies r>0.99", "KID normal", "KID pneumonia"], rows))
        md.append("\n*Low disagreement = synthetic images that are easy to classify. 'synth. > val p95' expected ≈ 5 %. "
                  "KID has its own scale: it is only useful for comparing across guidance values.*\n")
        if any("precision_normal" in c for c in quality):
            md.append("### Precision / recall of the distribution and epistemic uncertainty\n")
            pr = lambda c, cls: (f"{fmt(c.get(f'precision_{cls}'))} / {fmt(c.get(f'recall_{cls}'))} "
                                 f"(ref. {fmt(c.get(f'precision_ref_{cls}'))} / {fmt(c.get(f'recall_ref_{cls}'))})")
            rows = [[f"{c['guidance']:g}", c.get("n_seeds", "-"), pr(c, "normal"), pr(c, "pneumonia"),
                     f"{fmt(c.get('mi_synth'), 4)} ({fmt(c.get('mi_val'), 4)})"] for c in quality]
            md.append(md_table(["guidance", "seeds", "precision / recall normal (real val)",
                                "precision / recall pneumonia (real val)", "epistemic uncertainty synth. (real val)"], rows))
            md.append("\n*Precision = fidelity (synthetic inside the real support); recall = coverage (diversity). High precision with "
                      "recall well below the reference means mode collapse, which is what a high guidance tends to produce. "
                      "Computed in the verifier's feature space, like KID. Epistemic uncertainty is the mutual information "
                      "of MC-dropout (nats); the network only has dropout in its dense head, so it is a ranking tool, "
                      "not a calibrated estimate.*\n")

    # ---------- sondas ----------
    h1, h2, gaps, diag, verdicts = aggregate_probes()
    if h1:
        md.append("## 3. Shortcuts in the real images (H1): AUC of each probe\n")
        rows = [[probe, fmt_mean_sd(summarize(sc[SC_VAL])), fmt_mean_sd(summarize(sc[SC_TEST]))] for probe, sc in h1.items()]
        md.append(md_table(["probe", "AUC → val", "AUC → test"], rows))
        md.append("\nIf a poor part of the image (brightness, edge) already gives a high AUC, that part carries class signal.\n")
        save_csv([{"probe": p, "auc_val": summarize(e[SC_VAL])["mean"], "auc_test": summarize(e[SC_TEST])["mean"]}
                  for p, e in h1.items()], SUMMARY_DIR / "probes_h1.csv")
    if diag:
        md.append("### Diagnosis: false positives on normals and edge brightness\n")
        rows = [[subset, fmt_mean_sd(summarize([x[0] for x in v]), 1), fmt_mean_sd(summarize([x[1] for x in v]), 2)] for subset, v in diag.items()]
        md.append(md_table(["set", "% false positives", "rho(P, edge brightness)"], rows))
        md.append("")
    if h2:
        md.append("## 4. Does the DDPM amplify or mitigate the shortcut? (H2)\n")
        rows, csv_h2 = [], []
        for (w, probe), sc in sorted(h2.items()):
            r, sn = summarize(sc[SC_CV_REAL]), summarize(sc[SC_CV_SYNTH])
            v = verdicts[w][probe]
            rows.append([f"{w:g}", probe, fmt_mean_sd(r), fmt_mean_sd(sn),
                         f"{v['amplifies']} / {v['mitigates']} / {v['no_difference']}"])
            csv_h2.append({"guidance": w, "probe": probe, "auc_real": r["mean"], "auc_synthetic": sn["mean"],
                           "seeds_amplifies": v["amplifies"], "seeds_mitigates": v["mitigates"],
                           "seeds_no_difference": v["no_difference"]})
        md.append(md_table(["guidance", "probe", "real AUC (CV)", "synthetic AUC (CV)",
                            "seeds: amplifies / mitigates / no difference"], rows))
        save_csv(csv_h2, SUMMARY_DIR / "probes_h2.csv")
        md.append("\nEach seed decides using the intervals of its own cross-validation; with few synthetic images "
                  "'no difference' is the usual outcome and does not prove there is no difference.\n")
    if gaps:
        md.append("### Class separation in brightness (Cohen's d, pneumonia − normal)\n")
        rows = [[f"{w:g}", subset, measure, fmt_mean_sd(summarize(v), 2)] for (w, subset, measure), v in sorted(gaps.items())
                if subset in ("real 10%", "test", "synthetic")]
        md.append(md_table(["guidance", "set", "measure", "Cohen's d"], rows))
        md.append("\nIf the synthetic images separate the classes in brightness more than the real 10%, the generator exaggerates that difference.\n")

    # ---------- escenarios ----------
    data, ceiling = load_scenarios()
    rows_sc, values = aggregate_scenarios(data, ceiling)
    if rows_sc:
        save_csv(rows_sc, SUMMARY_DIR / "scenarios_summary.csv")
        with open(SUMMARY_DIR / "scenarios_summary.json", "w", encoding="utf-8") as f:
            json.dump(rows_sc, f, indent=2, ensure_ascii=False, default=float)
        md.append("## 5. Scenarios: difference versus base, across seeds\n")
        md.append("Each difference is **paired** (same subsample and same test) and computed per seed. "
                  "`*` marks that the 95% interval across seeds does not include 0. "
                  "'Seeds +' is in how many seeds the difference is positive. Δ AUC and Δ specificity come from the paired \n"
                  "bootstrap of each seed; Δ sensitivity is the per-seed difference of point estimates (same subsample, same test). \n"
                  "ECE is the calibration error: lower is better (its paired differences are in `scenarios_summary.csv`).\n")
        for (w, r) in sorted({(k[0], k[1]) for k in values}):
            md.append(f"### Guidance {w:g}, ratio ×{r:g}\n")
            rows, items_auc, items_spec, items_sens = [], [], [], []
            for e in ORDER:
                if (w, r, e) not in values:
                    continue
                v, res = values[(w, r, e)]
                da, ds, dse = res["d_auc"], res["d_spec"], res["d_sens"]
                common_cols = [e, res["auc"]["n"], fmt_mean_sd(res["auc"]), fmt_mean_sd(res["sens"]),
                               fmt_mean_sd(res["spec"]), fmt_mean_sd(res["ece"])]
                if e == BASE:
                    rows.append(common_cols + ["-", "-", "-", "-", "-"])
                    continue
                rows.append(common_cols + [
                    f"{fmt(da['mean'], 3, True)} [{fmt(da['ci_low'], 3, True)}, {fmt(da['ci_high'], 3, True)}]"
                    + (" *" if excludes_zero(da) else ""), f"{da['positive']}/{da['n']}",
                    f"{fmt(dse['mean'], 3, True)} [{fmt(dse['ci_low'], 3, True)}, {fmt(dse['ci_high'], 3, True)}]"
                    + (" *" if excludes_zero(dse) else ""),
                    f"{fmt(ds['mean'], 3, True)} [{fmt(ds['ci_low'], 3, True)}, {fmt(ds['ci_high'], 3, True)}]"
                    + (" *" if excludes_zero(ds) else ""), f"{ds['positive']}/{ds['n']}"])
                items_auc.append((e, v["d_auc"], da))
                items_spec.append((e, v["d_spec"], ds))
                items_sens.append((e, v["d_sens"], dse))
            md.append(md_table(["scenario", "seeds", "test AUC", "test sensitivity", "test specificity", "test ECE",
                                "Δ AUC vs base [CI]", "seeds +", "Δ sensitivity vs base [CI]",
                                "Δ specificity vs base [CI]", "seeds +"], rows))
            md.append("")
            if items_auc:
                tag = f"w{w:g}_x{r:g}"
                delta_figure(items_auc, f"AUC change versus base (guidance {w:g}, ×{r:g})",
                             "Δ AUC on test (mean of training replicates)", SUMMARY_DIR / f"delta_auc_{tag}.png")
                delta_figure(items_spec, f"Specificity change versus base (guidance {w:g}, ×{r:g})",
                             "Δ specificity on test (threshold 0.5)", SUMMARY_DIR / f"delta_spec_{tag}.png")
                delta_figure(items_sens, f"Sensitivity change versus base (guidance {w:g}, ×{r:g})",
                             "Δ sensitivity on test (threshold 0.5)", SUMMARY_DIR / f"delta_sens_{tag}.png")
            print(f"\n[guidance {w:g}, ratio ×{r:g}]")
            for row in rows:
                print("  " + " | ".join(str(c) for c in row))
        sel, corr = aggregate_selection()
        if sel:
            md.append("### What each filter actually selected\n")
            rows, csv_rows = [], []
            for (w, r) in sorted(sel):
                for e in ORDER:
                    if e not in sel[(w, r)]:
                        continue
                    g = sel[(w, r)][e]
                    dis, cf, mi = (summarize(g["pct_verifier_disagrees"]), summarize(g["mean_confidence"]),
                                   summarize(g.get("mean_mutual_info", [])))
                    rows.append([f"{w:g}", f"×{r:g}", e, dis["n"], fmt_mean_sd(dis, 1), fmt_mean_sd(cf), fmt_mean_sd(mi, 4)])
                    csv_rows.append({"guidance": w, "ratio": r, "scenario": e, "n_seeds": dis["n"],
                                     "pct_verifier_disagrees": dis["mean"], "mean_confidence": cf["mean"],
                                     "mean_epistemic": mi["mean"]})
            md.append(md_table(["guidance", "ratio", "scenario", "seeds", "% added where the verifier disagrees",
                                "mean confidence", "mean epistemic"], rows))
            md.append("\n*Only the added synthetic images. A filter that improves things while adding many images the verifier "
                      "disagrees with is not 'cleaning' anything: it is changing something else. The 'epistemic' filters "
                      "(E2d low, E2e high) use the mutual information of MC-dropout.*\n")
            save_csv(csv_rows, SUMMARY_DIR / "selection_diagnostics.csv")
        if corr:
            ws = sorted({w for (w, _) in corr})
            pairs = sorted({pair for (_, pair) in corr})
            rows = [[pair] + [fmt_mean_sd(summarize(list(corr[(w, pair)].values())), 2) if (w, pair) in corr else "-"
                              for w in ws] for pair in pairs]
            md.append("### Do the selection scores measure the same thing? (Spearman correlation)\n")
            md.append(md_table(["pair of scores"] + [f"guidance {w:g}" for w in ws], rows))
            md.append("\n*|ρ| = 1 means the two scores rank the images identically, so filtering by one or the other selects "
                      "the same images. Entropy and margin are redundant with the 'boundary' filter (E2c); the epistemic "
                      "score is the one that adds new information.*\n")
        n_max = max(res["auc"]["n"] for _, res in values.values())
        if n_max < 5:
            md.append(f"> **Note:** there are only {n_max} seed(s). With so few, the intervals across seeds are very "
                      "wide (or do not exist with 1). Run all 5 subsamples for firmer conclusions.\n")

    if not rows_sc and not ver and not quality and not h1 and not shift_md:
        raise SystemExit("No results found in outputs/. Run Pipeline.py first (or the individual scripts).")

    md.append("## Limitations to declare\n\n"
              "- Each seed is a different 10% subsample with its own generator and verifier; "
              "the differences are measured within each seed and then pooled.\n"
              "- The interval across seeds uses a Student t with few degrees of freedom: it is honest but wide.\n"
              "- AUC differences of 0.01–0.02 are usually indistinguishable from noise.\n"
              "- The verifier that selects the synthetic images was trained on the same real images, so the selection "
              "is biased toward what it already knows.\n"
              "- Without a patient identifier, patient-level leakage cannot be ruled out; a single dataset and a single generator.\n"
              "- KID is computed in the feature space of the verifier (a small CNN trained on ~470 images), not in a "
              "standard Inception space, so it only serves to compare guidance values within this project.\n"
              "- Memorization is measured with pixel correlation, not with distances between embeddings.\n"
              "- The number of images cannot be equalized in the synthetic-only scenario or in the ceiling; "
              "those two comparisons also mix in a different amount of data.\n"
              "- The domain classifier and the KS test describe the shift between train and test, but they do not prove "
              "which feature the models use; that is what the shortcut probes are for.\n")
    with open(SUMMARY_DIR / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nSummary saved in {SUMMARY_DIR} (summary.md, .csv tables and .png figures)")


if __name__ == "__main__":
    main()
