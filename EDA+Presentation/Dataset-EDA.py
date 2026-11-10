#!/usr/bin/env python3
"""
EDA rapido de PneumoniaMNIST (28x28) para elegir y justificar el dataset (hito a).

Uso (entorno AdvancedAITopics):
    python eda_pneumoniamnist_compact.py [--out eda_output] [--scarcity 0.1]

Genera en --out:
    overview.png           desbalance de clases + shift de brillo + clasificador de dominio
    samples.png            imagenes al azar por particion y clase
    mean_images.png        imagen media por clase y diferencia val/test frente a train
    closest_pairs_*.png    parejas de imagenes mas parecidas (candidatas a duplicados) con su diferencia
    shortcut_probes.png    AUC prediciendo la clase con informacion parcial (candidatos a atajo)
    *.csv                  las mismas tablas que se imprimen por pantalla
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # solo guarda ficheros, no abre ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from medmnist import PneumoniaMNIST
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPLITS = ("train", "val", "test")
CLASSES = ("normal", "pneumonia")  # etiqueta 0 = normal, 1 = pneumonia
PAIRS = (("train", "val"), ("train", "test"))  # comparamos val y test siempre contra train


def wilson(k, n, z=1.96):
    """Intervalo de confianza del 95 % de Wilson para una proporcion k/n (mejor que el normal con n pequeno)."""
    p, d = k / n, 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def domain_auc(xa, xb, seed):
    """AUC (validacion cruzada 5-fold) de un clasificador que intenta adivinar si una imagen es de `a` o de `b`.
    0.5 = no puede distinguirlas (sin shift detectable); cerca de 1 = las particiones son claramente distintas."""
    X, d = np.vstack([xa, xb]), np.r_[np.zeros(len(xa)), np.ones(len(xb))]
    model = make_pipeline(PCA(50, random_state=seed), StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return roc_auc_score(d, cross_val_predict(model, X, d, cv=cv, method="predict_proba")[:, 1])


def plot_samples(imgs, y, path, n=8, seed=0):
    """Imagenes al azar: una fila por (particion, clase). Sirve para ver a ojo como son los datos."""
    rng = np.random.default_rng(seed)
    rows = [(s, c) for s in SPLITS for c in (0, 1)]
    fig, ax = plt.subplots(len(rows), n, figsize=(1.5 * n, 1.6 * len(rows)))
    for i, (s, c) in enumerate(rows):
        for j, k in enumerate(rng.choice(np.flatnonzero(y[s] == c), n, replace=False)):
            ax[i, j].imshow(imgs[s][k], cmap="gray", vmin=0, vmax=255)
            ax[i, j].axis("off")
        ax[i, 0].text(-0.15, 0.5, f"{s}\n{CLASSES[c]}", transform=ax[i, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.suptitle("Random samples per split and class")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_closest_pairs(imgs, y, split, r, nn, path, k=6):
    """Las k imagenes de `split` cuya vecina mas parecida en train tiene mayor correlacion r.
    Fila 1: la imagen; fila 2: su vecina en train; fila 3: |diferencia| (negro = identicas)."""
    fig, ax = plt.subplots(3, k, figsize=(1.9 * k, 6.4))
    for j, i in enumerate(np.argsort(-r)[:k]):
        a, b = imgs[split][i].astype(int), imgs["train"][nn[i]].astype(int)
        ax[0, j].imshow(a, cmap="gray", vmin=0, vmax=255)
        ax[0, j].set_title(f"{split}[{i}] {CLASSES[y[split][i]]}", fontsize=8)
        ax[1, j].imshow(b, cmap="gray", vmin=0, vmax=255)
        ax[1, j].set_title(f"train[{nn[i]}] {CLASSES[y['train'][nn[i]]]}", fontsize=8)
        ax[2, j].imshow(np.abs(a - b), cmap="magma", vmin=0, vmax=255)
        ax[2, j].set_title(f"r={r[i]:.3f}  max|dif|={np.abs(a - b).max()}", fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    for i, label in enumerate((split, "nearest in train", "|difference|")):
        ax[i, 0].text(-0.1, 0.5, label, transform=ax[i, 0].transAxes, rotation=90, ha="right", va="center", fontsize=9)
    fig.suptitle(f"Most similar {split} -> train pairs (r = pixel correlation)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="carpeta donde medmnist descarga el dataset")
    ap.add_argument("--out", default="eda_output")
    ap.add_argument("--scarcity", type=float, default=0.10, help="fraccion del train que forma el subconjunto escaso S")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out, root = Path(args.out), Path(args.root)
    out.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    tables = {}  # nombre -> DataFrame; al final se guardan todos como CSV

    # ------------------------------------------------------------------ 1. Carga
    # imgs[s]: (N, 28, 28) uint8; y[s]: (N,) con 0 = normal y 1 = pneumonia.
    # X[s]: las mismas imagenes aplanadas (N, 784) en [0, 1], para los modelos y las correlaciones.
    imgs, y = {}, {}
    for s in SPLITS:
        ds = PneumoniaMNIST(split=s, download=True, root=str(root), size=28)
        imgs[s], y[s] = ds.imgs, ds.labels.ravel()
    X = {s: imgs[s].reshape(len(imgs[s]), -1).astype(np.float32) / 255 for s in SPLITS}

    # ------------------------------------------------------------------ 2. Desbalance de clases
    # Cuantas imagenes hay de cada clase en cada particion y si la proporcion cambia (label shift).
    # majority_class_acc = accuracy de un "clasificador" que siempre predice la clase mayoritaria:
    # es la referencia minima que cualquier modelo tiene que superar.
    rows = []
    for s in SPLITS:
        n, k = len(y[s]), int(y[s].sum())
        lo, hi = wilson(k, n)
        rows.append(dict(split=s, n=n, normal=n - k, pneumonia=k, pct_pneumonia=100 * k / n,
                         ci_low=100 * lo, ci_high=100 * hi, majority_class_acc=100 * max(k, n - k) / n))
    tables["class_balance"] = bal = pd.DataFrame(rows)
    print("\n== Class balance ==\n" + bal.round(2).to_string(index=False))
    # Chi-cuadrado: contrasta si la proporcion de clases es la misma en las dos particiones.
    for a, b in PAIRS:
        t = [[(y[s] == c).sum() for c in (0, 1)] for s in (a, b)]
        print(f"chi-square {a} vs {b}: p = {stats.chi2_contingency(t)[1]:.3g}")
    # Incertidumbre de medir con pocas imagenes: semianchura del IC 95 % de una accuracy de 0.85.
    for s in ("val", "test"):
        hw = 196 * np.sqrt(0.85 * 0.15 / len(y[s]))
        print(f"IC 95% de una accuracy de 0.85 en {s} (n={len(y[s])}): +/-{hw:.1f} pp")

    # ------------------------------------------------------------------ 3. Shift en las imagenes (KS)
    # Brillo (media de pixeles) y contraste (desviacion tipica) de cada imagen. El test KS compara la
    # distribucion de cada particion con la de train: ks_stat es la mayor distancia entre las dos curvas
    # acumuladas (0 = iguales, 1 = totalmente distintas). Con n grande el p-valor sale pequeno casi siempre:
    # interpreta sobre todo el estadistico. Se hace global y por clase para no confundir con el label shift.
    feat = pd.concat([pd.DataFrame(dict(split=s, cls=np.take(CLASSES, y[s]), brightness=X[s].mean(1), contrast=X[s].std(1)))
                      for s in SPLITS], ignore_index=True)
    rows = []
    for scope in ("all",) + CLASSES:
        sub = feat if scope == "all" else feat[feat.cls == scope]
        for a, b in PAIRS:
            for f in ("brightness", "contrast"):
                ks = stats.ks_2samp(sub.loc[sub.split == a, f], sub.loc[sub.split == b, f])
                rows.append(dict(scope=scope, pair=f"{a} vs {b}", feature=f, ks_stat=ks.statistic, p_value=ks.pvalue))
    tables["ks_tests"] = pd.DataFrame(rows)
    print("\n== KS tests (brillo y contraste por imagen) ==\n" + tables["ks_tests"].round(4).to_string(index=False))

    # ------------------------------------------------------------------ 4. Shift con un clasificador de dominio
    # Si un modelo sencillo distingue las imagenes de train de las de test, las particiones no vienen de la
    # misma distribucion. Hacerlo por clase aisla un cambio en las imagenes (P(x|y)) del cambio de prior.
    rows = []
    for scope in ("all",) + CLASSES:
        for a, b in PAIRS:
            m = {s: np.ones(len(y[s]), bool) if scope == "all" else y[s] == CLASSES.index(scope) for s in (a, b)}
            rows.append(dict(scope=scope, pair=f"{a} vs {b}", domain_auc=domain_auc(X[a][m[a]], X[b][m[b]], args.seed)))
    tables["domain_auc"] = dom = pd.DataFrame(rows)
    print("\n== Domain classifier AUC (0.5 = sin shift detectable) ==\n" + dom.round(3).to_string(index=False))

    # ------------------------------------------------------------------ 5. Fugas: duplicados y casi duplicados
    # (a) Imagenes identicas byte a byte entre particiones.
    h = {s: {im.tobytes() for im in imgs[s]} for s in SPLITS}
    print("\n== Duplicados exactos entre particiones ==")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        print(f"  {a} & {b}: {len(h[a] & h[b])}")
    # (b) Casi duplicados: para cada imagen, la correlacion de Pearson r con su vecina mas parecida en train.
    # Se centra y normaliza cada imagen, asi el producto escalar entre dos imagenes es directamente r.
    Z = {s: X[s] - X[s].mean(1, keepdims=True) for s in SPLITS}
    Z = {s: z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8) for s, z in Z.items()}
    near, rows = {}, []  # near[s] = (r, indice del vecino en train)
    print("== Vecino mas parecido en train (correlacion de Pearson r entre pixeles) ==")
    for s in SPLITS:
        c = Z[s] @ Z["train"].T  # r de cada imagen de `s` con cada imagen de train
        if s == "train":
            np.fill_diagonal(c, -1)  # en train-train se ignora la propia imagen
        near[s] = (c.max(1), c.argmax(1))
        rows.append(dict(split=s, median_r=np.median(near[s][0]), pct_r_ge_098=100 * (near[s][0] >= 0.98).mean()))
        print(f"  {s} -> train: r mediana = {rows[-1]['median_r']:.3f}; {rows[-1]['pct_r_ge_098']:.1f} % con vecino r >= 0.98")
    tables["near_duplicates"] = pd.DataFrame(rows)
    print("  Si val/test superan claramente a train -> train, hay casi duplicados (mismo paciente o adquisicion) entre particiones.")
    print("  MedMNIST no incluye id de paciente: la fuga por paciente no se puede descartar con estos datos.")

    # ------------------------------------------------------------------ 6. Escasez simulada
    # S = subconjunto estratificado del train (misma proporcion de clases) que simula tener pocos datos.
    idx, _ = train_test_split(np.arange(len(y["train"])), train_size=args.scarcity, stratify=y["train"], random_state=args.seed)
    np.save(out / f"scarce_train_indices_f{args.scarcity:g}_seed{args.seed}.npy", np.sort(idx))
    tables["scarcity"] = pd.DataFrame([dict(fraction=f, normal=int(round(f * (y["train"] == 0).sum())),
                                            pneumonia=int(round(f * (y["train"] == 1).sum())))
                                       for f in (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)])
    print("\n== Escasez simulada (imagenes de entrenamiento por clase) ==\n" + tables["scarcity"].to_string(index=False))
    print(f"Subconjunto S ({args.scarcity:.0%}): {len(idx)} imagenes, {100 * y['train'][idx].mean():.1f} % pneumonia "
          f"(train completo: {100 * y['train'].mean():.1f} %)")

    # ------------------------------------------------------------------ 7. Sondas de atajos (shortcut learning)
    # Se entrena un clasificador normal/neumonia con informacion parcial. Si algo no clinico (brillo y contraste
    # globales o el marco exterior de la imagen) ya predice bien la clase, hay un candidato a atajo.
    e = np.minimum(np.arange(28), 27 - np.arange(28))  # distancia de cada fila/columna al borde
    border = (np.minimum.outer(e, e) < 4).ravel()  # marco de 4 px: hombros, bordes, fondo
    probes = {"brightness + contrast": lambda s: np.c_[X[s].mean(1), X[s].std(1)],
              "border only (4 px frame)": lambda s: X[s][:, border],
              "center only": lambda s: X[s][:, ~border],
              "full image": lambda s: X[s]}
    rows = []
    for name, feats in probes.items():
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feats("train"), y["train"])
        rows.append(dict(probe=name, **{f"auc_{s}": roc_auc_score(y[s], m.predict_proba(feats(s))[:, 1]) for s in ("val", "test")}))
    tables["shortcut_probes"] = sp = pd.DataFrame(rows)
    print("\n== Sondas de atajos: AUC (entrenando en train) con informacion parcial ==\n" + sp.round(3).to_string(index=False))
    print("  AUC alto con solo brillo/contraste o solo el borde indica senal no pulmonar (candidato a atajo); no es una prueba.")
    print("  Si el AUC cae de val a test, esa senal no se mantiene en el test.")

    # ------------------------------------------------------------------ 8. Figuras
    # overview.png: (arriba izq.) imagenes por clase; (arriba der.) % de neumonia con su IC;
    # (abajo izq.) brillo medio por particion y clase; (abajo der.) AUC del clasificador de dominio.
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    long = bal.melt(id_vars="split", value_vars=list(CLASSES), var_name="class", value_name="count")
    sns.barplot(data=long, x="split", y="count", hue="class", ax=ax[0, 0])
    for c in ax[0, 0].containers:
        ax[0, 0].bar_label(c, padding=2)
    ax[0, 0].set_title("Images per class")
    ax[0, 1].errorbar(range(3), bal.pct_pneumonia, yerr=[bal.pct_pneumonia - bal.ci_low, bal.ci_high - bal.pct_pneumonia],
                      fmt="o", capsize=6)
    ax[0, 1].set(xticks=range(3), xticklabels=SPLITS, ylim=(0, 100), ylabel="% pneumonia (Wilson 95 % CI)",
                 title="Class prior per split")
    sns.violinplot(data=feat, x="split", y="brightness", hue="cls", hue_order=list(CLASSES), order=list(SPLITS),
                   cut=0, inner="quartile", ax=ax[1, 0])
    ax[1, 0].set_title("Mean brightness per image")
    sns.barplot(data=dom, x="pair", y="domain_auc", hue="scope", ax=ax[1, 1])
    ax[1, 1].axhline(0.5, color="k", ls="--", lw=1)
    ax[1, 1].set(ylim=(0.4, 1), title="Domain classifier AUC (0.5 = indistinguishable)", xlabel="")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=150)
    plt.close(fig)

    # samples.png: imagenes al azar por particion y clase.
    plot_samples(imgs, y, out / "samples.png")

    # mean_images.png: imagen media de cada clase en train y diferencia (val - train) y (test - train).
    # Rojo/azul = pixeles que en esa particion son mas/menos brillantes que en train.
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.5))
    for r, c in enumerate(CLASSES):
        mean = {s: imgs[s][y[s] == r].mean(0) for s in SPLITS}
        ax[r, 0].imshow(mean["train"], cmap="gray", vmin=0, vmax=255)
        ax[r, 0].set_title(f"Mean image, {c} (train)")
        for j, s in enumerate(("val", "test"), start=1):
            d = mean[s] - mean["train"]
            im = ax[r, j].imshow(d, cmap="RdBu_r", vmin=-abs(d).max(), vmax=abs(d).max())
            ax[r, j].set_title(f"{s} - train, {c}")
            plt.colorbar(im, ax=ax[r, j], fraction=0.046)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "mean_images.png", dpi=150)
    plt.close(fig)

    # closest_pairs_*.png: las parejas "supuestamente iguales" (mayor correlacion con una imagen de train).
    for s in SPLITS:
        plot_closest_pairs(imgs, y, s, *near[s], out / f"closest_pairs_{s}.png")

    # shortcut_probes.png: AUC de cada sonda en val y test (la linea discontinua es el azar).
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.barplot(data=sp.melt(id_vars="probe", var_name="split", value_name="auc"), y="probe", x="auc", hue="split", ax=ax)
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.set(xlim=(0.4, 1), ylabel="", title="Predicting the class from partial information (AUC)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "shortcut_probes.png", dpi=150)
    plt.close(fig)

    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    print(f"\nListo. Resultados en {out.resolve()}")


if __name__ == "__main__":
    main()