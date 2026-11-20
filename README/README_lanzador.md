## Running the Full Pipeline (`Lanzar_todo.bat`)

`Lanzar_todo.bat` launches the entire process with a double click or from the console: data cleaning, conditional DDPM, verifier, shortcut probes, scenarios, and a final summary across seeds. Under the hood, it calls `Pipeline.py`.

### Requirements

- Windows with the `AdvancedAITopics` conda environment (if yours has a different name, change the `set "ENTORNO=..."` line at the top of the `.bat`).
- PyTorch with GPU (CUDA) support. It can be run on CPU, but the full pipeline would take far longer.
- The dataset at `data\pneumoniamnist.npz` and these files in the project root: `Lanzar_todo.bat`, `Pipeline.py`, `Limpieza.py`, `DDPM.py`, `Verificador.py`, `Sondas.py`, `Escenarios.py`, `Agregar.py`, and `comun.py`.

### How to launch it

From the project root, in a console with conda (or by double-clicking the file):

```
Lanzar_todo.bat                 # shows a menu
Lanzar_todo.bat simular         # shows the commands without running anything
Lanzar_todo.bat rapido          # short test of the whole pipeline
Lanzar_todo.bat completo        # long run: 5 seeds, guidance 0 and 2
Lanzar_todo.bat agregar         # only merges the results that already exist
```

You can append `Pipeline.py` options at the end, for example:

```
Lanzar_todo.bat completo --semillas 0 1 2 --guidances 2
```

Recommended order: `simular` → `rapido` → `completo`.

### What each mode does

| Mode | What it runs | What it's for |
|---|---|---|
| `simular` | Nothing; only shows the commands | Review the plan before spending hours of GPU time |
| `rapido` | 1 seed, 3000-step DDPM, few replicates | Check that everything works and get a first result |
| `completo` | 5 seeds, guidance 0 and 2, 20000-step DDPM | Final project results |
| `agregar` | Only `Agregar.py` | Regenerate tables and figures without recomputing anything |

The full mode can take many hours, so it's best to launch it overnight. Disable your computer's sleep mode beforehand, because the script doesn't change that setting.

### If it gets interrupted

The process is **resumable**: run the same command again and it will continue where it left off, skipping steps that already have their result. To redo something from scratch, add `--forzar`:

```
Lanzar_todo.bat completo --forzar
```

### Where the results are

| Path | Contents |
|---|---|
| `data\limpio\` | Cleaned data, 10% subsamples, and the cleaning report |
| `outputs\ddpm\` | Trained models, control images, and generated synthetic images |
| `outputs\verificador\` | Verifiers and evaluation of the synthetic images (label noise, memorization, KID) |
| `outputs\sondas\` | Shortcut probes on real and synthetic images |
| `outputs\escenarios\` | Scenario comparison per seed |
| `outputs\resumen\` | **Final summary across seeds** (`resumen.md`), `.csv` tables, and figures |
| `outputs\logs\` | Messages from each run |

The main file for reading the results is `outputs\resumen\resumen.md`.

### Common problems

- **"Can't find conda"**: open the `.bat` from *Anaconda Prompt* with the environment activated, or add the path of your conda installation inside the file.
- **Warning that there is no GPU**: your PyTorch is the CPU version (`torch.__version__` ends in `+cpu`). Reinstall it with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/).
- **`OMP: Error #15`**: already handled in the scripts; if it appears again, check that you're using the current version of the `.py` files.
- **Slow with the project in OneDrive**: the pipeline saves many files (the models, every 2000 steps). Pause syncing or move the project to a local folder.
- **A step fails**: the process moves on to the next seed and records it in the final summary. Check `outputs\logs\` and rerun the same command.
