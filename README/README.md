# DDPM Augmentation under Data Scarcity, Shortcuts and Distribution Shift on PneumoniaMNIST

## Main idea

When labeled medical data is scarce, a common remedy is to augment the training set with synthetic images from a **diffusion model (DDPM)**. But if the real data contains **shortcuts** (spurious features such as brightness or border artifacts that correlate with the label) and the test set comes from a **shifted distribution**, synthetic data may not help: the generator learns the same shortcuts and could even exaggerate them.

This project tests that on PneumoniaMNIST. We simulate scarcity (10% of the cleaned training set), train a class-conditional DDPM on that small subset, and ask:

> Does DDPM augmentation inherit, amplify or mitigate shortcut dependence, and how does it affect generalization on a shifted test set?

### Hypotheses

| ID | Hypothesis |
|---|---|
| **H1** | Classifiers exploit shortcuts (brightness/contrast, border) present in the real data. |
| **H2** | The DDPM amplifies the shortcut: probes predict the class better on synthetic images than on real ones. |
| **H3** *(extra, not implemented yet)* | Generating synthetic images that **break** the shortcut-class correlation improves generalization on the shifted test set. |

---

## Workflow at a glance

1. **Protocol:** fix evaluation rules before touching the test set; multiple seeds and confidence intervals.
2. **Audit and clean the data:** remove duplicates (train↔train, val↔train).
3. **Characterize the shift** between the clean train and the official test (prior shift, domain classifier, KS, difference map).
4. **Simulate scarcity:** 10% stratified subsamples, at least 5 seeds; the other 90% is only the ceiling.
5. **Baseline classifier and shortcut probes (H1).**
6. **Class-conditional DDPM** trained only on the 10%, sampled at several guidance scales.
7. **Quality control** of the generated images.
8. **Uncertainty-based selection** of synthetic samples (confidence and epistemic uncertainty).
9. **Shortcut analysis on synthetic data (H2).**
10. **Compare scenarios** on the official test set, with paired statistics across seeds.

---

## Project structure

```
.
├── Run_all.bat          Windows launcher (menu or one-word modes)
├── Pipeline.py          Resumable orchestrator: seeds x guidance x ratios
├── common.py            Paths, seeds, data loading, pixel correlation
├── Cleaning.py          Phases 1-2: deduplication and 10% subsamples
├── Shift.py             Phase 1: train vs test shift characterization
├── DDPM.py              Phase 4: class-conditional DDPM + classifier-free guidance
├── Verifier.py          Phases 3 and 5-6: baseline CNN, quality control, uncertainty
├── Probes.py            Phases 3 and 7: shortcut probes on real (H1) and synthetic (H2) data
├── Scenarios.py         Phases 6, 8, 9: filters, scenarios and paired evaluation
├── Aggregate.py         Phase 9: statistics across seeds and summary report
├── data/
│   ├── pneumoniamnist.npz        original dataset (input)
│   └── clean/                    created by Cleaning.py
└── outputs/
    ├── shift/  ddpm/  verifier/  probes/  scenarios/    results of each stage
    ├── summary/                                          summary.md, tables and figures
    └── logs/                                             one log per pipeline run
```

| File | Phases | Main outputs |
|---|---|---|
| `Cleaning.py` | 1, 2 | `data/clean/pneumoniamnist_clean.npz`, `subsamples.npz`, `cleaning_report.json` |
| `Shift.py` | 1 | `outputs/shift/shift.json`, `shift_domain.csv`, `shift_ks.csv`, `shift_difference_map.png` |
| `DDPM.py` | 4 | `ddpm_seed{s}.pt`, `synthetic_seed{s}_w{w}.npz`, preview grids |
| `Verifier.py` | 3, 5, 6 | `verifier_seed{s}.pt`, `evaluation_synthetic_seed{s}_w{w}.json/.npz/.png` |
| `Probes.py` | 3, 7 | `probes_*.json/.csv`, `gaps_*.csv`, `h1_probes_*.png`, `h2_probes_*.png` |
| `Scenarios.py` | 6, 8, 9 | `scenarios_*.csv/.json`, `probabilities_*.npz`, `auc_*.png`, `sens_spec_*.png` |
| `Aggregate.py` | 9 | `outputs/summary/summary.md` plus CSV tables and figures |

---

## How to run

Requirements: Python with `torch`, `scikit-learn`, `scipy` and `matplotlib`; the original dataset at `data/pneumoniamnist.npz`. `Run_all.bat` expects a conda environment named `AdvancedAITopics` (editable at the top of the file).

```
Run_all.bat quick        short test of the whole pipeline (1 seed, a few minutes with GPU)
Run_all.bat full         full pipeline (5 seeds, guidance 0/2/5; many hours, best run overnight)
Run_all.bat shift        only the train vs test shift analysis (a few minutes)
Run_all.bat aggregate    only merge the existing results into the summary
Run_all.bat simulate     show the commands without running anything
```

Any `Pipeline.py` option can be appended, for example:

```
Run_all.bat full --seeds 0 1 2 --guidances 2
Run_all.bat full --ratios 1 2 5          # also runs the x2 and x5 synthetic/real ratios
Run_all.bat full --guidances 0 2         # previous setup, without guidance 5
```

The pipeline is **resumable**: if it is interrupted, launch the same command and it skips every step whose result already exists. `--force` redoes the selected steps and `--only` runs just some of them (`cleaning shift ddpm generate verifier evaluate probes scenarios aggregate`). Results produced before a script was extended are skipped too; to recompute them use `--force` with the corresponding `--only` step (for example `--only evaluate --force` is cheap, whereas redoing `scenarios` retrains every scenario).

The scripts can also be run one by one (`python Cleaning.py`, `python DDPM.py train --seed 0`, `python Verifier.py evaluate --seed 0 --synthetic ...`, and so on); each one documents its options in its header.

The final report is written to `outputs/summary/summary.md`.

---

## Phase details

### Phase 0. Protocol
With little data the variance of the risk estimate is high, and with a shifted test set ($Q \neq P$) the test risk is not the train risk. Evaluation rules are therefore fixed **before** looking at the test set, using several seeds/subsamples and confidence intervals.

### Phase 1. Data audit and cleaning
- **Deduplication** (`Cleaning.py`): SHA-1 hashing of the bytes for exact duplicates (copies with contradictory labels are dropped entirely) and pixel correlation r > 0.99 for near-duplicates. Validation is cleaned against the train set and against itself; the test set is **never modified**, only audited. Duplicates cause leakage: validation AUC ≈ 0.98 versus ≈ 0.92 on test is the symptom.
- **Shift characterization** (`Shift.py`, run once, independent of the seed), clean train versus official test:
  - *Prior shift* ($P(y) \neq Q(y)$): pneumonia drops from 74% to 62.5%. Reported with 95% intervals, the test−train difference and the importance weights $p_{test}(y)/p_{train}(y)$.
  - *Covariate shift* ($P(x) \neq Q(x)$): test−train mean-difference map per class, plus its profile by distance to the border.
  - *Domain classifier*: per class and per region (whole image, border ring, center). AUC ≈ 0.5 means indistinguishable domains; ≈ 0.78 on normals points to class-conditional shift. Every AUC comes with a **null AUC** (train split at random, same sizes) and with train vs val as a reference.
- **Kolmogorov-Smirnov:** report the statistic $D$ with a bootstrap CI, not the p-value (with large *n* everything is "significant"), together with a permutation **noise floor**. Computed on brightness, contrast, border brightness and center brightness. KS is univariate, so it is complemented with the domain classifier.

### Phase 2. Scarcity simulation
10% of the deduplicated train set, stratified by class, repeated over at least 5 seeds. The remaining 90% is used only as the **ceiling** (all real data). The subsample's own shift is measured with the domain classifier (`domain_auc_vs_rest` in `cleaning_report.json`).

### Phase 3. Baseline classifier and shortcut probes (H1)
- **Classifier** (`VerifierCNN`): small CNN of three convolutional blocks, BCE loss weighted by class, early stopping on a **clean** validation set; AUC plus per-class sensitivity/specificity.
- **Probes** that see only partial information, trained on the real 10% with a heavily regularized logistic regression: brightness and contrast only; mean border brightness (2 px); border only (frame of *k* = 1, 2, 4 px); center only (without a 4 px frame); and the full image as a reference. A high-AUC border probe is evidence of a shortcut.
- **Diagnostic:** check whether false positives on test-normal correlate with border brightness (Spearman ρ and false-positive rate by border-brightness tercile), linking the shortcut to real failures.
- *Caution:* with 28×28 images a 4 px frame covers a lot of anatomy, so thinner frames are tested too.

### Phase 4. DDPM
- Forward process: $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ with a cosine noise schedule and $T = 1000$; a U-Net $\epsilon_\theta(x_t,t,c)$ (28×28 → 14×14 → 7×7, attention at the bottleneck) predicts the noise, trained with $\mathcal{L}=\mathbb{E}\big[\|\epsilon-\epsilon_\theta\|^2\big]$ and weights averaged with an EMA.
- **Class conditioning** $c\in\{\text{normal},\text{pneumonia},\varnothing\}$ with $p_{drop}= 0.15$, and **classifier-free guidance** $\tilde\epsilon=(1+w)\,\epsilon_\theta(x_t,t,c)-w\,\epsilon_\theta(x_t,t,\varnothing)$. A high $w$ yields more typical, less diverse samples, which matters for shortcuts. The pipeline samples a grid of guidance scales, **$w \in \{0, 2, 5\}$**, where $w = 5$ is deliberately high to test whether the shortcut gets amplified.
- Sampling is a fast deterministic DDIM-style sampler (100 steps by default); the full 1000-step DDPM sampler is available.
- A compact DDPM from scratch (LoRA would only apply to a large pretrained model).
- **Critical rule:** the generator never sees the test set or the other 90% of train.

### Phase 5. Quality control of generated images
- **Memorization:** correlation with the nearest neighbor in the real 10%, compared with the same quantity for real validation images (expected ≈ 5% of synthetic images above the validation 95th percentile). Critical with ~470 images.
- **Fidelity and diversity:**
  - **KID** (unbiased) is preferred over FID with few samples.
  - **Distribution precision / recall** ($k = 3$): precision is fidelity (synthetic inside the real support), recall is coverage (diversity). High precision with a recall well below the reference signals mode collapse.
  - Both are computed in the feature space of the verifier, with real validation versus real train as a reference.
- **Label noise:** disagreement between requested and predicted labels under the baseline classifier, compared with the classifier's own error on real validation images.

### Phase 6. Uncertainty-based selection
Uncertainty splits into aleatoric (irreducible) and epistemic (reducible). Scores computed for every synthetic image:
- **Confidence in the requested class**, predictive **entropy** and top-2 **margin**. In a binary classifier entropy and margin are monotone in $|p-0.5|$, so they rank images exactly like the *boundary* filter and are not run as separate scenarios; the Spearman correlation between scores is reported to show it.
- **Epistemic uncertainty:** mutual information of **MC-dropout** (30 passes). The network only has dropout in its dense head, so this is the head's uncertainty, useful to rank images but not a calibrated estimate.

Filters compared: **high-confidence** (*easy*) vs *medium* vs **low-confidence** (*boundary*) vs *worst* (negative control) vs **random**, plus **low / high epistemic uncertainty**. Very easy samples add little, very hard ones may be noise, and the useful ones are usually in between. For every filter the summary also reports **what it actually selected** (share of images where the verifier disagrees with the requested class, mean confidence, mean epistemic uncertainty). Because the filtering classifier is trained on the same ~470 real images, it is biased toward what it already knows, so it is used only as a verifier while the DDPM is conditioned by class.

### Phase 7. Shortcut analysis on synthetic data (H2)
Repeat the Phase 3 probes on the synthetic images (stratified cross-validation inside the real 10% and inside the synthetic set, plus cross-transfer real ↔ synthetic). If brightness/contrast or the border predict the class **better** than on real images (non-overlapping 95% intervals), the generator **amplifies** the shortcut; if **worse**, it **mitigates** it; otherwise there is no clear difference. FID/KID cannot detect this, as they measure global similarity rather than which features become predictive.

### Phase 8. Experimental scenarios

| Scenario | Training data |
|---|---|
| Baseline | 10% real |
| Classical aug. | 10% real + transformed copies (rotation, scale, shift, brightness/contrast) |
| E1 | 10% real + random synthetic |
| E2a | 10% real + *easy* synthetic (high confidence in the requested class) |
| E2b | 10% real + *medium* synthetic |
| E2c | 10% real + *boundary* synthetic (the verifier doubts) |
| E2d | 10% real + synthetic with **low** epistemic uncertainty |
| E2e | 10% real + synthetic with **high** epistemic uncertainty |
| E3 | 10% real + worse synthetic (negative control) |
| E4 | Synthetic only |
| H3 (extra) | 10% real + synthetic that breaks the shortcut-class correlation *(not implemented yet)* |
| Ceiling | 100% deduplicated real |

**Controls:** same number of added images across all scenarios that add data, including the classical augmentation (otherwise any gain may just be "more data"; E4 and the ceiling cannot match it by construction); vary the synthetic/real ratio (×1 by default, `--ratios 1 2 5` for ×2 and ×5); each scenario is trained several times with different initializations; always evaluate on the **official test set**, and separately on test-normal and test-pneumonia (sensitivity and specificity).

### Phase 9. Evaluation and statistics
- **Metrics:** AUC, per-class sensitivity/specificity, ECE.
- **Intervals:** paired bootstrap over the 624 test images (the same resamples for every scenario) plus variance across seeds and subsamples.
- **Comparisons:** paired differences per seed with CIs rather than isolated p-values. ΔAUC and Δspecificity come from the paired bootstrap of each seed; Δsensitivity and ΔECE are per-seed differences of point estimates. Across seeds, the mean difference is reported with a Student-*t* 95% interval, a `*` when it excludes 0, and in how many seeds it is positive. AUC differences of 0.01–0.02 may be indistinguishable from noise, and this should be stated plainly.

---

## Limitations

- Without patient identifiers, patient-level leakage among near-duplicates cannot be ruled out.
- The filtering classifier shares data with the baseline evaluation classifier.
- A single dataset and a single generator.
- Epistemic uncertainty comes from MC-dropout on the dense head only; an ensemble of verifiers would be more faithful.
- KID, precision/recall and the verifier's features live in a space learned from ~470 images, not in a standard Inception space, so they are only meaningful for comparing guidance values inside this project. Memorization uses pixel correlation, not embedding distances.
- The shortcut probes are linear models; a small CNN on the border or the center could find non-linear signal they miss.
- E4 and the ceiling do not use the same number of images as the other scenarios.
- The domain classifier and KS describe the shift between train and test; they do not prove which feature a model actually uses. That is what the shortcut probes are for.
- H3 is not implemented yet.
