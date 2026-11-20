"""
Pipeline.py
===========
Lanza el flujo completo para varias semillas (submuestras del 10 %) y varios valores
de guidance, uno detrás de otro, para poder dejarlo corriendo sin vigilarlo.

Para cada semilla hace, en este orden:
    1. Limpieza.py            (una sola vez para todo el proyecto)
    2. DDPM.py entrenar       generador entrenado solo con el 10 % de esa semilla
    3. Verificador.py entrenar clasificador base (también sirve para elegir sintéticas)
    4. Sondas.py              sondas de atajo sobre las reales (H1)
    y, para cada valor de guidance:
    5. DDPM.py generar        imágenes sintéticas
    6. Verificador.py evaluar ruido de etiquetas, memorización y KID
    7. Sondas.py              sondas sobre las sintéticas (H2)
    8. Escenarios.py          comparación de escenarios (por cada --ratio)
y al final:
    9. Agregar.py             junta todas las semillas (diferencias pareadas con intervalos)

Es REANUDABLE: si un paso ya tiene su resultado, se salta. Si se corta la luz o cierras
la ventana, vuelve a lanzar el mismo comando y continúa por donde iba. Con --forzar se
rehace todo lo seleccionado.

Uso:
    # Primera prueba rápida (1 semilla, entrenamientos cortos, unos minutos con GPU)
    python Pipeline.py --rapido

    # Ver qué haría sin ejecutar nada
    python Pipeline.py --dry-run

    # Flujo largo completo (5 semillas, guidance 0 y 2, 20000 pasos): déjalo de noche
    python Pipeline.py

    # Solo algunos pasos
    python Pipeline.py --solo sondas escenarios agregar

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

from comun import DIR_SALIDAS, RAIZ, RUTA_LIMPIO, RUTA_SUBMUESTRAS

PASOS = ["limpieza", "ddpm", "generar", "verificador", "evaluar", "sondas", "escenarios", "agregar"]

DIR_DDPM = DIR_SALIDAS / "ddpm"
DIR_VERIF = DIR_SALIDAS / "verificador"
DIR_SONDAS = DIR_SALIDAS / "sondas"
DIR_ESC = DIR_SALIDAS / "escenarios"
DIR_LOGS = DIR_SALIDAS / "logs"


# ==========================================================================
# Ejecutar un script y apuntar cómo ha ido
# ==========================================================================
class Ejecutor:
    def __init__(self, dry_run: bool, log):
        self.dry_run, self.log = dry_run, log
        self.resumen = []  # (nombre, estado, segundos)

    def saltar(self, nombre, motivo="ya existe"):
        print(f"\n--- {nombre}: saltado ({motivo})")
        self.resumen.append((nombre, "saltado", 0.0))

    def correr(self, nombre, script, argumentos) -> bool:
        cmd = [sys.executable, str(RAIZ / script), *[str(a) for a in argumentos]]
        mostrar = " ".join(f'"{c}"' if " " in c else c for c in [script, *[str(a) for a in argumentos]])
        print(f"\n>>> {nombre}\n    python {mostrar}")
        self.log.write(f"\n>>> {nombre}\n    python {mostrar}\n")
        if self.dry_run:
            self.resumen.append((nombre, "(simulado)", 0.0))
            return True
        t0 = time.time()
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        with subprocess.Popen(cmd, cwd=RAIZ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                              text=True, encoding="utf-8", errors="replace") as p:
            for linea in p.stdout:
                print(linea, end="")
                self.log.write(linea)
            self.log.flush()
        ok = p.returncode == 0
        dt = time.time() - t0
        self.resumen.append((nombre, "ok" if ok else "FALLÓ", dt))
        if not ok:
            print(f"!!! {nombre} ha fallado (código {p.returncode}). Se continúa con la siguiente semilla.")
        return ok


def paso_ddpm_hecho(semilla: int, pasos: int) -> bool:
    """El modelo existe Y se entrenó al menos 'pasos' pasos (una prueba corta anterior no cuenta)."""
    ruta = DIR_DDPM / f"ddpm_semilla{semilla}.pt"
    if not ruta.exists():
        return False
    try:
        import torch
        return int(torch.load(ruta, map_location="cpu")["paso"]) >= pasos
    except Exception:
        return False


def sinteticas_hechas(ruta, n_por_clase: int) -> bool:
    """El archivo existe y tiene al menos el número de imágenes pedido."""
    if not ruta.exists():
        return False
    try:
        return len(np.load(ruta)["labels"]) >= 2 * n_por_clase
    except Exception:
        return False


def limpieza_hecha(semillas) -> bool:
    if not (RUTA_LIMPIO.exists() and RUTA_SUBMUESTRAS.exists()):
        return False
    claves = set(np.load(RUTA_SUBMUESTRAS).files)
    return all(f"idx_10pct_semilla{s}" in claves for s in semillas)


# ==========================================================================
# Programa principal
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="Flujo completo para varias semillas y valores de guidance")
    ap.add_argument("--semillas", type=int, nargs="+", default=None, help="submuestras del 10 %% (por defecto 0 1 2 3 4)")
    ap.add_argument("--guidances", type=float, nargs="+", default=None, help="valores de guidance (por defecto 0 2)")
    ap.add_argument("--ratios", type=float, nargs="+", default=[1.0], help="sintéticas añadidas = ratio x nº de reales")
    ap.add_argument("--pasos-ddpm", type=int, default=None, help="pasos de entrenamiento del DDPM (por defecto 20000)")
    ap.add_argument("--ddpm-canales", type=int, default=64, help="anchura de la U-Net (32 si vas justo)")
    ap.add_argument("--n-por-clase", type=int, default=None, help="sintéticas a generar por clase (por defecto, las necesarias, mín. 1000)")
    ap.add_argument("--pasos-muestreo", type=int, default=100, help="pasos al generar (100 = rápido; 1000 = DDPM completo)")
    ap.add_argument("--verif-epocas", type=int, default=100)
    ap.add_argument("--reps", type=int, default=None, help="entrenamientos por escenario (por defecto 3)")
    ap.add_argument("--epocas", type=int, default=None, help="épocas máximas de los clasificadores de escenarios (por defecto 60)")
    ap.add_argument("--bootstrap", type=int, default=None, help="remuestreos de los intervalos (por defecto 1000)")
    ap.add_argument("--escenarios", nargs="+", default=None, help="escenarios a evaluar (por defecto todos)")
    ap.add_argument("--solo", nargs="+", choices=PASOS, default=None, help="ejecutar solo estos pasos")
    ap.add_argument("--forzar", action="store_true", help="rehacer aunque el resultado ya exista")
    ap.add_argument("--rapido", action="store_true", help="primera prueba corta de todo el flujo")
    ap.add_argument("--dry-run", action="store_true", help="solo muestra los comandos, no ejecuta nada")
    args = ap.parse_args()

    # ---------- valores por defecto (normal o rápido) ----------
    def v(x, rapido, normal):
        return x if x is not None else (rapido if args.rapido else normal)

    semillas = v(args.semillas, [0], [0, 1, 2, 3, 4])
    guidances = [float(w) for w in v(args.guidances, [2.0], [0.0, 2.0])]
    pasos_ddpm = v(args.pasos_ddpm, 3000, 20000)
    reps = v(args.reps, 2, 3)
    epocas = v(args.epocas, 30, 60)
    bootstrap = v(args.bootstrap, 300, 1000)
    todos = ["base", "clasica", "e1", "e2a", "e2b", "e2c", "e3", "e4", "techo"]
    escenarios = v(args.escenarios, [e for e in todos if e != "techo"], todos)
    activos = set(args.solo or PASOS)

    # ---------- cuántas sintéticas hacen falta ----------
    n_real = 469  # estimación (~10 % del train limpio); se afina si ya existe la limpieza
    if limpieza_hecha(semillas):
        try:
            from comun import cargar_submuestra
            n_real = len(cargar_submuestra(semillas[0])[0])
        except Exception:
            pass
    necesarias = int(math.ceil(max(args.ratios) * n_real))
    n_por_clase = v(args.n_por_clase, 100, max(1000, necesarias))
    if n_por_clase < necesarias:
        print(f"AVISO: con --ratio {max(args.ratios):g} hacen falta unas {necesarias} sintéticas por clase y generarás {n_por_clase}. "
              "Los filtros usarán casi todo el pool y se parecerán entre sí.")

    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    log_ruta = DIR_LOGS / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    print("=" * 78)
    print(f"Semillas: {semillas} | guidance: {guidances} | ratios: {args.ratios}")
    print(f"DDPM: {pasos_ddpm} pasos | sintéticas por clase: {n_por_clase} | reps: {reps} | épocas: {epocas} | bootstrap: {bootstrap}")
    print(f"Escenarios: {escenarios}")
    print(f"Pasos activos: {sorted(activos, key=PASOS.index)} | forzar: {args.forzar} | dry-run: {args.dry_run}")
    print(f"Log: {log_ruta}")
    print("=" * 78)

    t_global = time.time()
    fallos = []
    with open(log_ruta, "w", encoding="utf-8") as log:
        ex = Ejecutor(args.dry_run, log)

        # ---------- 1) limpieza (una vez) ----------
        if "limpieza" in activos:
            if limpieza_hecha(semillas) and not args.forzar:
                ex.saltar("limpieza")
            else:
                ok = ex.correr("limpieza", "Limpieza.py", ["--k", max(semillas) + 1])
                if not ok:
                    sys.exit("Sin limpieza no se puede continuar.")

        # ---------- por semilla ----------
        for s in semillas:
            print("\n" + "#" * 78 + f"\n# SEMILLA {s}\n" + "#" * 78)
            ok = True

            if "ddpm" in activos and ok:
                if paso_ddpm_hecho(s, pasos_ddpm) and not args.forzar:
                    ex.saltar(f"s{s} ddpm entrenar", f"ya entrenado con >= {pasos_ddpm} pasos")
                else:
                    ok = ex.correr(f"s{s} ddpm entrenar", "DDPM.py",
                                   ["entrenar", "--semilla", s, "--pasos", pasos_ddpm, "--canales", args.ddpm_canales,
                                    "--vista-cada", 2000])

            if "verificador" in activos and ok:
                if (DIR_VERIF / f"verificador_semilla{s}.pt").exists() and not args.forzar:
                    ex.saltar(f"s{s} verificador entrenar")
                else:
                    ok = ex.correr(f"s{s} verificador entrenar", "Verificador.py",
                                   ["entrenar", "--semilla", s, "--epocas", args.verif_epocas])

            if "sondas" in activos and ok:
                if (DIR_SONDAS / f"sondas_semilla{s}.json").exists() and not args.forzar:
                    ex.saltar(f"s{s} sondas reales (H1)")
                else:
                    ok = ex.correr(f"s{s} sondas reales (H1)", "Sondas.py", ["--semilla", s, "--bootstrap", bootstrap])

            techo_hecho = False
            for w in guidances:
                if not ok:
                    break
                stem = f"sinteticas_semilla{s}_w{w}"
                npz = DIR_DDPM / f"{stem}.npz"
                etiqueta = stem.replace("sinteticas_", "")

                if "generar" in activos:
                    if sinteticas_hechas(npz, n_por_clase) and not args.forzar:
                        ex.saltar(f"s{s} w{w:g} generar")
                    else:
                        ok = ex.correr(f"s{s} w{w:g} generar", "DDPM.py",
                                       ["generar", "--semilla", s, "--n-normal", n_por_clase, "--n-neumonia", n_por_clase,
                                        "--guidance", w, "--pasos-muestreo", args.pasos_muestreo])
                        if not ok:
                            break

                if "evaluar" in activos:
                    if (DIR_VERIF / f"evaluacion_{stem}.json").exists() and not args.forzar:
                        ex.saltar(f"s{s} w{w:g} evaluar sintéticas")
                    else:
                        ok = ex.correr(f"s{s} w{w:g} evaluar sintéticas", "Verificador.py",
                                       ["evaluar", "--semilla", s, "--sinteticas", npz])
                        if not ok:
                            break

                if "sondas" in activos:
                    if (DIR_SONDAS / f"sondas_{etiqueta}.json").exists() and not args.forzar:
                        ex.saltar(f"s{s} w{w:g} sondas (H2)")
                    else:
                        ok = ex.correr(f"s{s} w{w:g} sondas (H2)", "Sondas.py",
                                       ["--semilla", s, "--sinteticas", npz, "--bootstrap", bootstrap])
                        if not ok:
                            break

                if "escenarios" in activos:
                    for r in args.ratios:
                        # el techo no depende de las sintéticas: se entrena una sola vez por semilla
                        lista = [e for e in escenarios if not (e == "techo" and techo_hecho)]
                        if "techo" in lista:
                            techo_hecho = True
                        salida = DIR_ESC / f"escenarios_{etiqueta}_x{r:g}.json"
                        nombre = f"s{s} w{w:g} x{r:g} escenarios"
                        if salida.exists() and not args.forzar:
                            ex.saltar(nombre)
                            continue
                        ok = ex.correr(nombre, "Escenarios.py",
                                       ["--semilla", s, "--sinteticas", npz, "--ratio", r, "--reps", reps,
                                        "--epocas", epocas, "--bootstrap", bootstrap, "--escenarios", *lista])
                        if not ok:
                            break
                    if not ok:
                        break

            if not ok:
                fallos.append(s)

        # ---------- agregar ----------
        if "agregar" in activos:
            ex.correr("agregar semillas", "Agregar.py", [])

        # ---------- resumen ----------
        print("\n" + "=" * 78 + "\nRESUMEN")
        for nombre, estado, seg in ex.resumen:
            t = f"{seg / 60:6.1f} min" if seg else ""
            print(f"  {nombre:<40s} {estado:<10s} {t}")
        print(f"Tiempo total: {(time.time() - t_global) / 60:.1f} min")
        if fallos:
            print(f"Semillas con algún paso fallido: {fallos}. Revisa el log y relanza el mismo comando para reanudar.")
            sys.exit(1)


if __name__ == "__main__":
    main()
