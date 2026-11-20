## Running the Full Pipeline (`Run_all.bat`)

`Run_all.bat` launches the entire process with a double click or from the console: data cleaning, conditional DDPM, verifier, shortcut probes, scenarios, and a final summary across seeds. Under the hood, it calls `Pipeline.py`.

### Requirements

- Windows with the `AdvancedAITopics` conda environment (if yours has a different name, change the `set "ENV_NAME=..."` line at the top of the `.bat`).
- PyTorch with GPU (CUDA) support. It can be run on CPU, but the full pipeline would take far longer.
- The dataset at `data\pneumoniamnist.npz` and these files in the project root: `Run_all.bat`, `Pipeline.py`, `Cleaning.py`, `DDPM.py`, `Verifier.py`, `Probes.py`, `Scenarios.py`, `Aggregate.py`, and `common.py`.

### How to launch it

From the project root, in a console with conda (or by double-clicking the file):

```
Run_all.bat                 # shows a menu
Run_all.bat simulate        # shows the commands without running anything
Run_all.bat quick           # short test of the whole pipeline
Run_all.bat full            # long run: 5 seeds, guidance 0 and 2
Run_all.bat aggregate       # only merges the results that already exist
```

You can append `Pipeline.py` options at the end, for example:

```
Run_all.bat full --seeds 0 1 2 --guidances 2
```

Recommended order: `simulate` → `quick` → `full`.

### What each mode does

| Mode | What it runs | What it's for |
|---|---|---|
| `simulate` | Nothing; only shows the commands | Review the plan before spending hours of GPU time |
| `quick` | 1 seed, 3000-step DDPM, few replicates | Check that everything works and get a first result |
| `full` | 5 seeds, guidance 0 and 2, 20000-step DDPM | Final project results |
| `aggregate` | Only `Aggregate.py` | Regenerate tables and figures without recomputing anything |

The full mode can take many hours, so it's best to launch it overnight. Disable your computer's sleep mode beforehand, because the script doesn't change that setting.

### If it gets interrupted

The process is **resumable**: run the same command again and it will continue where it left off, skipping steps that already have their result. To redo something from scratch, add `--force`:

```
Run_all.bat full --force
```

### Where the results are

| Path | Contents |
|---|---|
| `data\clean\` | Cleaned data, 10% subsamples, and the cleaning report |
| `outputs\ddpm\` | Trained models, preview images, and generated synthetic images |
| `outputs\verifier\` | Verifiers and evaluation of the synthetic images (label noise, memorization, KID) |
| `outputs\probes\` | Shortcut probes on real and synthetic images |
| `outputs\scenarios\` | Scenario comparison per seed |
| `outputs\summary\` | **Final summary across seeds** (`summary.md`), `.csv` tables, and figures |
| `outputs\logs\` | Messages from each run |

The main file for reading the results is `outputs\summary\summary.md`.

### Common problems

- **"Cannot find conda"**: open the `.bat` from *Anaconda Prompt* with the environment activated, or add the path of your conda installation inside the file.
- **Warning that there is no GPU**: your PyTorch is the CPU version (`torch.__version__` ends in `+cpu`). Reinstall it with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/).
- **`OMP: Error #15`**: already handled in the scripts; if it appears again, check that you're using the current version of the `.py` files.
- **Slow with the project in OneDrive**: the pipeline saves many files (the models, every 2000 steps). Pause syncing or move the project to a local folder.
- **A step fails**: the process moves on to the next seed and records it in the final summary. Check `outputs\logs\` and rerun the same command.
- **Old results from the Spanish version**: file names, folders and checkpoint keys have changed (e.g. `outputs\resumen\` is now `outputs\summary\`), so previous models and `data\limpio\` are not reused. Rerun the pipeline from the start (or rename the old files by hand).
