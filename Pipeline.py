"""
Pipeline.py
===========
Lanza el flujo completo para varias semillas (submuestras del 10 %) y varios valores
de guidance, uno detrás de otro, para poder dejarlo corriendo sin vigilarlo.

Para cada semilla hace, en este orden:
    1. Cleaning.py             (una sola vez para todo el proyecto)
    1b. Shift.py               desplazamiento train vs test: prevalencia, clasificador de
                               dominio, KS y mapa de diferencias (una sola vez, no depende de la semilla)
    2. DDPM.py train           generador entrenado solo con el 10 % de esa semilla
    3. Verifier.py train       clasificador base (también sirve para elegir sintéticas)
    4. Probes.py               sondas de atajo sobre las reales (H1)
    y, para cada valor de guidance:
    5. DDPM.py generate        imágenes sintéticas
    6. Verifier.py evaluate    ruido de etiquetas, memorización y KID
    7. Probes.py               sondas sobre las sintéticas (H2)
    8. Scenarios.py            comparación de escenarios (por cada --ratios)
y al final:
    9. Aggregate.py            junta todas las semillas (diferencias pareadas con intervalos)

Es REANUDABLE: si un paso ya tiene su resultado, se salta. Si se corta la luz o cierras
la ventana, vuelve a lanzar el mismo comando y continúa por donde iba. Con --force se
rehace todo lo seleccionado.

Uso:
    # Primera prueba rápida (1 semilla, entrenamientos cortos, unos minutos con GPU)
    python Pipeline.py --quick

    # Ver qué haría sin ejecutar nada
    python Pipeline.py --dry-run

    # Flujo largo completo (5 semillas, guidance 0, 2 y 5, 20000 pasos): déjalo de noche
    # (para el flujo anterior, sin guidance 5:  python Pipeline.py --guidances 0 2)
    python Pipeline.py

    # Solo algunos pasos
    python Pipeline.py --only probes scenarios aggregate

    # Solo el análisis de desplazamiento train vs test
    python Pipeline.py --only shift

    # Más ratios de sintéticas (×1, ×2, ×5): necesita más sintéticas, se calcula solo
    python Pipeline.py --ratios 1 2 5

Los mensajes de cada paso se guardan en outputs/logs/.
"""

import os
# En Windows + conda, NumPy y PyTorch a veces cargan cada uno su copia de OpenMP y
# Python se cierra con "OMP: Error #15". Esta línea lo evita. Debe ir ANTES de importar nada más.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import math
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

from common import OUTPUT_DIR, ROOT, CLEAN_PATH, SUBSAMPLES_PATH

STEPS = ["cleaning", "shift", "ddpm", "generate", "verifier", "evaluate", "probes", "scenarios", "aggregate"]

SHIFT_DIR = OUTPUT_DIR / "shift"
DDPM_DIR = OUTPUT_DIR / "ddpm"
VERIFIER_DIR = OUTPUT_DIR / "verifier"
PROBES_DIR = OUTPUT_DIR / "probes"
SCENARIOS_DIR = OUTPUT_DIR / "scenarios"
LOGS_DIR = OUTPUT_DIR / "logs"


# ==========================================================================
# Ejecutar un script y apuntar cómo ha ido
# ==========================================================================
class Runner:
    def __init__(self, dry_run: bool, log):
        self.dry_run, self.log = dry_run, log
        self.summary = []  # (nombre, estado, segundos)

    def skip(self, name, reason="already exists"):
        print(f"\n--- {name}: skipped ({reason})")
        self.summary.append((name, "skipped", 0.0))

    def run(self, name, script, arguments) -> bool:
        cmd = [sys.executable, str(ROOT / script), *[str(a) for a in arguments]]
        shown = " ".join(f'"{c}"' if " " in c else c for c in [script, *[str(a) for a in arguments]])
        print(f"\n>>> {name}\n    python {shown}")
        self.log.write(f"\n>>> {name}\n    python {shown}\n")
        if self.dry_run:
            self.summary.append((name, "(simulated)", 0.0))
            return True
        t0 = time.time()
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                              text=True, encoding="utf-8", errors="replace") as p:
            for line in p.stdout:
                print(line, end="")
                self.log.write(line)
            self.log.flush()
        ok = p.returncode == 0
        dt = time.time() - t0
        self.summary.append((name, "ok" if ok else "FAILED", dt))
        if not ok:
            print(f"!!! {name} failed (code {p.returncode}). Continuing with the next seed.")
        return ok


def ddpm_step_done(seed: int, steps: int) -> bool:
    """El modelo existe Y se entrenó al menos 'steps' pasos (una prueba corta anterior no cuenta)."""
    path = DDPM_DIR / f"ddpm_seed{seed}.pt"
    if not path.exists():
        return False
    try:
        import torch
        return int(torch.load(path, map_location="cpu")["step"]) >= steps
    except Exception:
        return False


def synthetic_done(path, n_per_class: int) -> bool:
    """El archivo existe y tiene al menos el número de imágenes pedido."""
    if not path.exists():
        return False
    try:
        return len(np.load(path)["labels"]) >= 2 * n_per_class
    except Exception:
        return False


def cleaning_done(seeds) -> bool:
    if not (CLEAN_PATH.exists() and SUBSAMPLES_PATH.exists()):
        return False
    keys = set(np.load(SUBSAMPLES_PATH).files)
    return all(f"idx_10pct_seed{s}" in keys for s in seeds)


# ==========================================================================
# Programa principal
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Full pipeline for several seeds and guidance values")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="10%% subsamples (default 0 1 2 3 4)")
    ap.add_argument("--guidances", type=float, nargs="+", default=None, help="guidance values (default 0 2 5; 5 is high on purpose to test whether the shortcut gets amplified)")
    ap.add_argument("--ratios", type=float, nargs="+", default=[1.0], help="synthetic added = ratio x number of real")
    ap.add_argument("--ddpm-steps", type=int, default=None, help="DDPM training steps (default 20000)")
    ap.add_argument("--ddpm-channels", type=int, default=64, help="U-Net width (32 if tight on resources)")
    ap.add_argument("--n-per-class", type=int, default=None, help="synthetic images to generate per class (default: as many as needed, min. 1000)")
    ap.add_argument("--sampling-steps", type=int, default=100, help="steps when generating (100 = fast; 1000 = full DDPM)")
    ap.add_argument("--verifier-epochs", type=int, default=100)
    ap.add_argument("--reps", type=int, default=None, help="trainings per scenario (default 3)")
    ap.add_argument("--epochs", type=int, default=None, help="max epochs of the scenario classifiers (default 60)")
    ap.add_argument("--bootstrap", type=int, default=None, help="resamples for the intervals (default 1000)")
    ap.add_argument("--scenarios", nargs="+", default=None, help="scenarios to evaluate (default all)")
    ap.add_argument("--only", nargs="+", choices=STEPS, default=None, help="run only these steps")
    ap.add_argument("--force", action="store_true", help="redo even if the result already exists")
    ap.add_argument("--quick", action="store_true", help="first short test of the whole pipeline")
    ap.add_argument("--dry-run", action="store_true", help="only shows the commands, runs nothing")
    args = ap.parse_args()

    # ---------- valores por defecto (normal o rápido) ----------
    def pick(x, quick, normal):
        return x if x is not None else (quick if args.quick else normal)

    seeds = pick(args.seeds, [0], [0, 1, 2, 3, 4])
    guidances = [float(w) for w in pick(args.guidances, [2.0], [0.0, 2.0, 5.0])]
    ddpm_steps = pick(args.ddpm_steps, 3000, 20000)
    reps = pick(args.reps, 2, 3)
    epochs = pick(args.epochs, 30, 60)
    bootstrap = pick(args.bootstrap, 300, 1000)
    all_scenarios = ["base", "classic", "e1", "e2a", "e2b", "e2c", "e2d", "e2e", "e3", "e4", "ceiling"]
    scenarios = pick(args.scenarios, [e for e in all_scenarios if e != "ceiling"], all_scenarios)
    active = set(args.only or STEPS)

    # ---------- cuántas sintéticas hacen falta ----------
    n_real = 469  # estimación (~10 % del train limpio); se afina si ya existe la limpieza
    if cleaning_done(seeds):
        try:
            from common import load_subsample
            n_real = len(load_subsample(seeds[0])[0])
        except Exception:
            pass
    needed = int(math.ceil(max(args.ratios) * n_real))
    n_per_class = pick(args.n_per_class, 100, max(1000, needed))
    if n_per_class < needed:
        print(f"WARNING: with --ratios {max(args.ratios):g} about {needed} synthetic images per class are needed and you will generate {n_per_class}. "
              "The filters will use almost the whole pool and will look alike.")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    print("=" * 78)
    print(f"Seeds: {seeds} | guidance: {guidances} | ratios: {args.ratios}")
    print(f"DDPM: {ddpm_steps} steps | synthetic per class: {n_per_class} | reps: {reps} | epochs: {epochs} | bootstrap: {bootstrap}")
    print(f"Scenarios: {scenarios}")
    print(f"Active steps: {sorted(active, key=STEPS.index)} | force: {args.force} | dry-run: {args.dry_run}")
    print(f"Log: {log_path}")
    print("=" * 78)

    t_global = time.time()
    failures = []
    with open(log_path, "w", encoding="utf-8") as log:
        ex = Runner(args.dry_run, log)

        # ---------- 1) limpieza (una vez) ----------
        if "cleaning" in active:
            if cleaning_done(seeds) and not args.force:
                ex.skip("cleaning")
            else:
                ok = ex.run("cleaning", "Cleaning.py", ["--k", max(seeds) + 1])
                if not ok:
                    sys.exit("Cannot continue without cleaning.")

        # ---------- 1b) desplazamiento train vs test (una vez; no depende de la semilla) ----------
        if "shift" in active:
            if (SHIFT_DIR / "shift.json").exists() and not args.force:
                ex.skip("shift (train vs test)")
            else:
                ex.run("shift (train vs test)", "Shift.py", ["--bootstrap", bootstrap])

        # ---------- por semilla ----------
        for s in seeds:
            print("\n" + "#" * 78 + f"\n# SEED {s}\n" + "#" * 78)
            ok = True

            if "ddpm" in active and ok:
                if ddpm_step_done(s, ddpm_steps) and not args.force:
                    ex.skip(f"s{s} ddpm train", f"already trained with >= {ddpm_steps} steps")
                else:
                    ok = ex.run(f"s{s} ddpm train", "DDPM.py",
                                ["train", "--seed", s, "--steps", ddpm_steps, "--channels", args.ddpm_channels,
                                 "--preview-every", 2000])

            if "verifier" in active and ok:
                if (VERIFIER_DIR / f"verifier_seed{s}.pt").exists() and not args.force:
                    ex.skip(f"s{s} verifier train")
                else:
                    ok = ex.run(f"s{s} verifier train", "Verifier.py",
                                ["train", "--seed", s, "--epochs", args.verifier_epochs])

            if "probes" in active and ok:
                if (PROBES_DIR / f"probes_seed{s}.json").exists() and not args.force:
                    ex.skip(f"s{s} real probes (H1)")
                else:
                    ok = ex.run(f"s{s} real probes (H1)", "Probes.py", ["--seed", s, "--bootstrap", bootstrap])

            ceiling_done = False
            for w in guidances:
                if not ok:
                    break
                stem = f"synthetic_seed{s}_w{w}"
                npz = DDPM_DIR / f"{stem}.npz"
                label = stem.replace("synthetic_", "")

                if "generate" in active:
                    if synthetic_done(npz, n_per_class) and not args.force:
                        ex.skip(f"s{s} w{w:g} generate")
                    else:
                        ok = ex.run(f"s{s} w{w:g} generate", "DDPM.py",
                                    ["generate", "--seed", s, "--n-normal", n_per_class, "--n-pneumonia", n_per_class,
                                     "--guidance", w, "--sampling-steps", args.sampling_steps])
                        if not ok:
                            break

                if "evaluate" in active:
                    if (VERIFIER_DIR / f"evaluation_{stem}.json").exists() and not args.force:
                        ex.skip(f"s{s} w{w:g} evaluate synthetic")
                    else:
                        ok = ex.run(f"s{s} w{w:g} evaluate synthetic", "Verifier.py",
                                    ["evaluate", "--seed", s, "--synthetic", npz])
                        if not ok:
                            break

                if "probes" in active:
                    if (PROBES_DIR / f"probes_{label}.json").exists() and not args.force:
                        ex.skip(f"s{s} w{w:g} probes (H2)")
                    else:
                        ok = ex.run(f"s{s} w{w:g} probes (H2)", "Probes.py",
                                    ["--seed", s, "--synthetic", npz, "--bootstrap", bootstrap])
                        if not ok:
                            break

                if "scenarios" in active:
                    for r in args.ratios:
                        # el techo no depende de las sintéticas: se entrena una sola vez por semilla
                        lst = [e for e in scenarios if not (e == "ceiling" and ceiling_done)]
                        if "ceiling" in lst:
                            ceiling_done = True
                        out = SCENARIOS_DIR / f"scenarios_{label}_x{r:g}.json"
                        name = f"s{s} w{w:g} x{r:g} scenarios"
                        if out.exists() and not args.force:
                            ex.skip(name)
                            continue
                        ok = ex.run(name, "Scenarios.py",
                                    ["--seed", s, "--synthetic", npz, "--ratio", r, "--reps", reps,
                                     "--epochs", epochs, "--bootstrap", bootstrap, "--scenarios", *lst])
                        if not ok:
                            break
                    if not ok:
                        break

            if not ok:
                failures.append(s)

        # ---------- agregar ----------
        if "aggregate" in active:
            ex.run("aggregate seeds", "Aggregate.py", [])

        # ---------- resumen ----------
        print("\n" + "=" * 78 + "\nSUMMARY")
        for name, status, sec in ex.summary:
            t = f"{sec / 60:6.1f} min" if sec else ""
            print(f"  {name:<40s} {status:<10s} {t}")
        print(f"Total time: {(time.time() - t_global) / 60:.1f} min")
        if failures:
            print(f"Seeds with a failed step: {failures}. Check the log and rerun the same command to resume.")
            sys.exit(1)


if __name__ == "__main__":
    main()
