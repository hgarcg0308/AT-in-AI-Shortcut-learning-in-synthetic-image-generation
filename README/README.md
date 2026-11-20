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
| **H3** *(extra)* | Generating synthetic images that **break** the shortcut-class correlation improves generalization on the shifted test set. |

---

## Workflow at a glance

1. **Protocol:** fix evaluation rules before touching the test set; multiple seeds and confidence intervals.
2. **Audit and clean the data:** remove duplicates (train↔train, val↔train) and characterize the shift.
3. **Simulate scarcity:** 10% stratified subsamples, at least 5 seeds; the other 90% is only the ceiling.
4. **Baseline classifier and shortcut probes (H1).**
5. **Class-conditional DDPM** trained only on the 10%.
6. **Quality control** of the generated images.
7. **Uncertainty-based selection** of synthetic samples.
8. **Shortcut analysis on synthetic data (H2).**
9. **Compare scenarios** on the official test set, with paired statistics.

---

## Phase details

### Phase 0. Protocol
With little data the variance of the risk estimate is high, and with a shifted test set ($Q \neq P$) the test risk is not the train risk. Evaluation rules are therefore fixed **before** looking at the test set, using several seeds/subsamples and confidence intervals.

### Phase 1. Data audit and cleaning
- **Deduplication:** byte hashing for exact duplicates; pixel correlation or embedding distance (e.g. r > 0.99) for near-duplicates. Duplicates cause leakage: validation AUC ≈ 0.98 versus ≈ 0.92 on test is the symptom.
- **Shift characterization:**
  - *Covariate shift* ($P(x) \neq Q(x)$): test−train difference map.
  - *Prior shift* ($P(y) \neq Q(y)$): pneumonia drops from 74% to 62.5%.
  - *Domain classifier*: AUC ≈ 0.5 means indistinguishable domains; ≈ 0.78 on normals points to class-conditional shift.
- **Kolmogorov-Smirnov:** report the statistic $D$ with a bootstrap CI, not the p-value, and complement it with the domain classifier since KS is univariate.

### Phase 2. Scarcity simulation
10% of the deduplicated train set, stratified by class, repeated over at least 5 seeds. The remaining 90% is used only as the **ceiling** (all real data). The subsample's own shift is measured with the domain classifier.

### Phase 3. Baseline classifier and shortcut probes (H1)
- **Classifier:** small CNN or adapted ResNet-18, BCE loss, early stopping on a **clean** validation set; AUC plus per-class sensitivity/specificity.
- **Probes** that see only partial information: brightness and contrast only; border only (frame of *k* = 1, 2, 4 px); center only. A high-AUC border probe is evidence of a shortcut.
- **Diagnostic:** check whether false positives on test-normal correlate with border brightness, linking the shortcut to real failures.
- *Caution:* with 28×28 images a 4 px frame covers a lot of anatomy, so thinner frames are worth testing.

### Phase 4. DDPM
- Forward process: $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$; a U-Net $\epsilon_\theta(x_t,t,c)$ predicts the noise, trained with $\mathcal{L}=\mathbb{E}\big[\|\epsilon-\epsilon_\theta\|^2\big]$.
- **Class conditioning** $c\in\{\text{normal},\text{pneumonia},\varnothing\}$ with $p_{drop}\approx 0.1$–$0.15$, and **classifier-free guidance** $\tilde\epsilon=(1+w)\,\epsilon_\theta(x_t,t,c)-w\,\epsilon_\theta(x_t,t,\varnothing)$. A high $w$ yields more typical, less diverse samples, which matters for shortcuts.
- A compact DDPM from scratch (LoRA would only apply to a large pretrained model).
- **Critical rule:** the generator never sees the test set or the other 90% of train.

### Phase 5. Quality control of generated images
- **Memorization:** nearest-neighbor distance to train versus the val→train distance distribution. Critical with ~470 images.
- **Fidelity and diversity:** prefer **KID** (unbiased) over FID with few samples.
- **Label noise:** disagreement between requested and predicted labels under the baseline classifier.

### Phase 6. Uncertainty-based selection
Uncertainty splits into aleatoric (irreducible) and epistemic (reducible). Measures: predictive entropy, top-2 margin, ensemble / MC-dropout disagreement. Compare **high-confidence** vs **low-confidence** vs **random** selection. Very easy samples add little, very hard ones may be noise, and the useful ones are usually in between. Because the filtering classifier is trained on the same ~470 real images, it is biased toward what it already knows, so it is used only as a verifier while the DDPM is conditioned by class.

### Phase 7. Shortcut analysis on synthetic data (H2)
Repeat the Phase 3 probes on the synthetic images. If brightness/contrast or the border predict the class **better** than on real images, the generator **amplifies** the shortcut; if **worse**, it **mitigates** it. FID/KID cannot detect this, as they measure global similarity rather than which features become predictive.

### Phase 8. Experimental scenarios

| Scenario | Training data |
|---|---|
| Baseline | 10% real |
| Classical aug. | 10% real + rotations, brightness, etc. |
| E1 | 10% real + all synthetic |
| E2 | 10% real + filtered synthetic (uncertainty criterion) |
| E3 | 10% real + worse synthetic (negative control) |
| E4 | Synthetic only |
| H3 (extra) | 10% real + synthetic that breaks the shortcut-class correlation |
| Ceiling | 100% deduplicated real |

**Controls:** same total number of images across scenarios (otherwise any gain may just be "more data"); vary the synthetic/real ratio (×1, ×2, ×5); always evaluate on the **official test set**, and separately on test-normal and test-pneumonia.

### Phase 9. Evaluation and statistics
- **Metrics:** AUC, per-class sensitivity/specificity, ECE.
- **Intervals:** bootstrap over the 624 test images plus variance across seeds and subsamples.
- **Comparisons:** paired differences per seed with CIs rather than isolated p-values. AUC differences of 0.01–0.02 may be indistinguishable from noise, and this should be stated plainly.

---

## Limitations

- Without patient identifiers, patient-level leakage among near-duplicates cannot be ruled out.
- The filtering classifier shares data with the baseline evaluation classifier.
- A single dataset and a single generator.

