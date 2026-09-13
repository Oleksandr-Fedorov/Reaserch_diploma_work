# PROJECT BRIEFING v6: Image Manipulation Detector — Master's Thesis
### (Context transfer — supersedes `PROJECT_BRIEFING_v5.md`. Read fully before advising.)

This document consolidates everything resolved since v5: a source-aware ELA retrain (closing a validation-leakage concern), a matching Fusion retrain, a full audit of an external AI's code-review claims (some confirmed, some refuted by checking actual code), a refutation of a false "ELA carries no signal" hypothesis via a from-scratch Decision Fusion experiment (including two failed/invalid attempts that are documented as cautionary methodology notes), and a de-facto four-way comparison. Do not propose changing the topic. Do not propose architecture changes or new datasets/signals without checking Section 18 (guardrails) first.

---

## 1. WHO I AM

Master's student in Computer Science. Committee: applied mathematicians, microcontroller/acoustics/ballistics specialists — not deep-ML experts. Narrative clarity and honestly-obtained, calibrated results matter more than implementation minutiae.

**Psychological note, updated again:** the target behavior from v5 ("verify once, act, move on" instead of serial opinion-shopping) held up well this phase, and a new, related good habit emerged: **treating a second AI's claims as hypotheses to check against actual code/numbers, not as facts to accept.** Concrete instances this phase:
- A second AI's claim that Fusion showed "practically no improvement over RGB" (implicitly RGB≈Fusion≈74%) was checked against the actual logged numbers and found false (RGB is 68%, not 74% — see Section 12). Three of that AI's five hypotheses were built on this false premise and were set aside as a result, rather than investigated further.
- A second AI's claim of a `filename` dictionary-collision risk across CASIA1/CASIA2 was checked against the actual generator code and found impossible by construction (see Section 11).
- A first attempt at a "properly val-tuned" Decision Fusion result (val macro F1 = 0.90, better than every other result in the project) was correctly flagged as suspicious *before* being reported, root-caused to a joint-grid multiple-comparisons overfit, and redone with a stricter two-stage search (Section 13) — this is the target pattern: noticing an anomalously-good number is itself a signal to double check, not a result to celebrate.

Keep reinforcing this pattern. No signs of regression to either serial opinion-shopping or uncritical acceptance of external AI claims.

---

## 2. TOPIC (FIXED — DO NOT SUGGEST CHANGING)

Image manipulation detector with localization and explainability. Three components: (1) ELA module — hand-implemented, non-ML; (2) CNN classifier; (3) GradCAM explainability. **Current phase:** both branches (RGB, ELA) and both fusion strategies (feature-level MLP fusion, decision-level probability fusion) now have clean, source-aware, leak-checked results on the in-domain (CASIA1+CASIA2) test set. The in-domain "does fusion help" question is now empirically answered (Section 14). The single largest remaining gap is cross-dataset generalization (Section 16, item 1) — untouched since v4.

---

## 3. FOUNDATIONAL WORK (COMPLETE — unchanged from v4/v5)

- Hand-implemented ELA (`ela.py`); strong signal on differing compression histories, near-zero signal on uniformly-resaved composites (the "bird-in-garden" case).
- Early CASIA1-only baseline: raw RGB CNN (~51%, no signal) vs. ELA+ResNet18 full fine-tune (86% acc, F1 0.85). The ~51% RGB number reflects that early experiment's setup, not RGB's ceiling — see Section 7.
- GradCAM confirmed the network attends to splice boundaries, not object interior.
- FastAPI + frontend prototype ("AEGIS // VISION") exists. **Still carries hardcoded placeholder sub-metrics and fake EXIF fields — unresolved, a defense risk if raised late.**

---

## 4. THE QUANTIZATION RESONANCE FINDING (established, unchanged)

V1 (fixed `q_ela=75`) showed a sharp confidence collapse exactly where `Q_primary` matched the hardcoded ELA reference quality — a property of the JPEG-Ghosts-style delta math, not data narrowness. The x4/Kaggle model (randomized `q_ela` during training) raised the resonance floor substantially (worst-case accuracy ~63% vs. V1's near-chance collapse) but did not eliminate the resonance dip itself. Full sweep results archived in `sweep_metrics_results.csv`; Q=85 wins on average, Q=95 wins on stability/floor — stated as an explicit trade-off in the thesis, not a single "best" answer.

---

## 5. ELA PHASE — CLOSED (unchanged conclusion; independently re-verified this phase — see Section 9)

Sufficient evidence at appropriate scale supports the resonance limitation, the augmentation-driven improvement, and the server-quality trade-off. This conclusion is not being re-opened. Note: Section 9 below describes a full ELA *retrain* on a stricter validation split — this was a data-integrity check, not a re-opening of the resonance conclusion, and it reconfirmed rather than changed the existing test-set numbers.

---

## 6. RGB DATASET PAIRING BUG — FOUND AND FIXED (unchanged from v5)

**The bug:** the original RGB-branch generator and the ELA generator were meant to apply the *same* primary JPEG quality (`q_primary`) per augmented instance, so a filename-matched RGB/ELA pair would represent the same compression scenario viewed two ways. They didn't. Root cause was two compounding issues, both in the RGB generator:

1. It re-sampled a fresh `q_primary` via `random.sample()` instead of reusing the value already fixed by the ELA generation run.
2. It preserved raw `os.listdir()` file order when filtering by the split map, while the ELA generator's `train_test_split(shuffle=True)` had reshuffled that order — so even the file *processing sequence* differed between the two scripts, on top of the RNG-state drift from unequal numbers of `random.sample()` calls per file.

**Severity, empirically confirmed:** a simulation reproducing both scripts' exact logic showed a 10.8% match rate on `q_primary` between filename-paired files — statistically indistinguishable from the ~9.1% expected under full decorrelation (11 possible quality values). Effectively every pair was scrambled, not just a "drifts after image 1" issue.

**Blast radius:** did NOT affect the ELA-only model, the train/test split integrity, or the RGB-only model's validity *as a standalone result* (RGB-only training never reads ELA data). DID invalidate every Late Fusion training/eval run done before the fix.

**Fix:** rebuilt the RGB generator (`RGB_v2`) to stop sampling entirely — it reads `filename`, `split`, `class`, and `q_primary` directly from the ELA log's CSV and replays them exactly, guaranteeing pairing by construction rather than by coincidence.

**Verification:** row-count, `q_primary`, and `split` diffed to zero mismatches between the two logs, confirmed directly (not just simulated).

---

## 7. RGB-ONLY BASELINE — RETRAINED ON v2, RESULT LOCKED IN (unchanged from v5)

Full fine-tune ResNet18 (ImageNet-pretrained), source-aware split (grouped by `original_filename` alone — see Section 11 for a noted asymmetry vs. ELA's composite-key grouping), leak check passing.

- Early stopped at epoch 5, best checkpoint = epoch 1 (val loss 0.6354).
- Test set (11,432 images), threshold = default 0.5:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| REAL | 0.71 | 0.76 | 0.73 |
| FAKE | 0.64 | 0.57 | 0.60 |

Accuracy 68%, macro F1 0.67. Checkpoint: `model_resnet_rgb_LATEST.pth`. This is the RGB number used throughout every comparison below — **not** 74% (see Section 12 for why this distinction matters).

---

## 8. LATE FUSION V1 — FIRST CLEAN RESULT (unchanged from v5; now superseded by V2, kept as footnote)

Built on the pre-source-aware ELA checkpoint. BatchNorm-freeze bug fixed (`train()` override forces both branches to `.eval()` unconditionally regardless of outer mode), correct RGB/ELA pairing via `RGB_v2`, macro-F1 threshold search (not FAKE-only F1, which had earlier produced a lopsided 0.61/0.90 REAL/FAKE recall split).

- Train 41,124 / Val 4,568 / Test 11,432 (source-aware, leak-free).
- Early stopped epoch 7, best checkpoint epoch 3 (val loss 0.6321). Threshold (macro F1) = 0.2610.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| REAL | 0.87 | 0.67 | 0.75 |
| FAKE | 0.65 | 0.86 | 0.74 |

Accuracy 75%, macro F1 0.75, **AUC 0.806**. Checkpoint: `model_fusion_LATEST.pth`.

**Status: superseded by V2 (Section 10) as the primary reported result.** Kept only as a documented "before the ELA source-aware retrain" data point — the V1→V2 delta is itself a useful methodology footnote (Section 10).

---

## 9. ELA RETRAIN — SOURCE-AWARE SPLIT (V2) — RESOLVED THIS PHASE

**Motivation:** an external AI reviewing the original ELA training script flagged that it used `random_split()` directly on the augmented-instance dataset (4 augmentations per source image), meaning "sibling" augmentations of the same source image could land on opposite sides of the train/val split. This is **augmentation leakage in validation**, not in test — the test set was always isolated at the raw-source-image level from the very first generator script (`train_test_split` runs before any augmentation is generated). The concern was specifically that val-based checkpoint selection might have been distorted by near-duplicate leakage, not that the test number itself was compromised.

**Action taken:** rewrote the ELA training script to use the same composite-key source-aware split logic already used for RGB, but keyed on `(source_dataset, original_filename)` (read from `dataset_log.csv`) instead of `original_filename` alone:
1. Build `file_to_source` dict mapping each augmented filename → `(source_dataset, original_filename)`.
2. Collect the set of unique `(source_dataset, original_filename)` tuples, sort them, shuffle with `random.Random(42)`.
3. Take the first 10% as validation sources; every augmented instance whose source tuple falls in that set goes to val, the rest to train.
4. Explicit leak check: `intersection = train_sources_check & val_sources_check` must be empty (asserted).
5. Split manifest saved to `ela_sourceaware_split_v2.csv` with columns `source_dataset, original_filename, subset` — full reproducibility.

**Leak check result:** `Train: 41124 | Val: 4568`, `Унікальних оригіналів Train: 10281 | Val: 1142`, **`Перетинів оригіналів: 0`** — confirmed leak-free.

**Training result:** ResNet18 full fine-tune, same recipe as always (AdamW, lr=5e-5, weight_decay=1e-4, label_smoothing=0.05, ReduceLROnPlateau, early stopping patience=4). Early stopped at epoch 5, best checkpoint = **epoch 1** (val loss 0.6037) — train loss kept dropping every subsequent epoch (0.52→0.40→0.28→0.19) while val loss rose (0.61→0.69→0.85→0.86), a clean, expected overfitting curve.

**Test metrics** (threshold = default 0.5, n = 11,432):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| REAL | 0.72 | 0.78 | 0.75 |
| FAKE | 0.66 | 0.58 | 0.61 |

Accuracy 69%, macro F1 0.68.

**Comparison to the old (non-source-aware) ELA result:** virtually identical (old was ~70% acc / ~0.69 macro F1; new is 69% / 0.68). **This is a positive, reassuring finding, not a regression** — since the test set was always properly isolated regardless of how validation was split, the near-identical test result empirically confirms the original ELA number was never artificially inflated by the validation-leakage issue. The leakage could only ever have subtly affected *which* checkpoint got selected as "best," not the reported test metric's validity.

**Suggested thesis framing:** present this as a proactive integrity check that produced a reassuring null result — "we identified a potential validation methodology weakness, retrained under a stricter protocol, and confirmed the original result was not an artifact of it." This is a strong, self-contained methodology-rigor paragraph for a committee that values honestly-obtained results over polish.

**Checkpoint:** `model_resnet_ela_sourceaware_v2.pth`. **Manifest:** `ela_sourceaware_split_v2.csv`. **Test predictions (with probabilities):** `ela_sourceaware_test_predictions.csv` (columns: `filename, true_label, pred_label, prob_fake`).

---

## 10. FUSION V2 — RETRAINED ON THE SOURCE-AWARE ELA BRANCH

Rebuilt using the new ELA checkpoint (`model_resnet_ela_sourceaware_v2.pth`) in place of the old one, with the RGB checkpoint (`model_resnet_rgb_LATEST.pth`) unchanged. Same frozen-backbone architecture, same BatchNorm-freeze `train()` override, same macro-F1 threshold search protocol as V1.

**Methodological caveat (documented, not a bug):** two things changed simultaneously between V1 and V2, not one — (a) the ELA branch is now the source-aware retrain, and (b) `label_smoothing=0.05` was added to the fusion head's `CrossEntropyLoss` in V2 (V1's fusion-head training did not have this). This means the V1→V2 delta cannot be cleanly attributed to "the better ELA branch" alone. Given the delta is tiny (see below), this is not a practical concern, but if this comparison is written up as a controlled before/after, it should be described honestly as "two simultaneous changes" rather than a single isolated variable.

**A consequence of the label_smoothing addition:** V2's val loss values (0.44–0.48) are **not directly comparable** to V1's (0.63–0.75) — label smoothing changes the CrossEntropy loss scale itself, this is not evidence of a "better-fitting" model. Only test-set accuracy / macro F1 / AUC are safely comparable across V1 and V2; val loss magnitudes are not.

**Split:** same composite-key source-aware logic as Section 9, applied to the Fusion dataset directly (train/val leak check passed, 0 intersection, Train 41,124 / Val 4,568 / Test 11,432).

**Training:** early stopped at epoch 7, best checkpoint at epoch 3.

**Threshold (macro F1 on val):** 0.2893 (V1 was 0.2610 — similar ballpark, a modest shift expected from the branch change).

**Test metrics** (n = 11,432):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| REAL | 0.89 | 0.62 | 0.73 |
| FAKE | 0.63 | 0.90 | 0.74 |

Accuracy 74%, macro F1 0.74.

**AUC** (confirmed from the saved ROC plot image, not printed in the console log): **0.802** (V1 was 0.806). Delta = 0.004 — within the normal single-seed noise band for a test set of this size (standard error on AUC at n≈11,432 is roughly ±0.003–0.005). **Not a meaningful regression.**

**Conclusion:** retraining Fusion on the more rigorously-validated ELA branch reproduces the original finding essentially unchanged. **Fusion V2 is now the primary reported result going forward.** V1 (75% / 0.75 macro F1 / AUC 0.806) is retained only as a documented "before the ELA integrity fix" footnote, per the existing guidance in v5 Section 8.

**Checkpoint:** `model_fusion_sourceaware_v2.pth`. **Metadata:** `fusion_v2_test_metadata.csv`. **Re-exported embeddings** (in a `fusion_v2/` output folder): `test_embeddings_rgb_512d.npy`, `test_embeddings_ela_512d.npy`, `test_embeddings_fused_1024d.npy`, `test_labels.npy`.

---

## 11. EXTERNAL CODE REVIEW — CLAIMS CHECKED AGAINST ACTUAL CODE

An external AI reviewed the split/generator/Fusion-V2 scripts and raised several points. Per the psychological note (Section 1), each claim was checked against the actual code rather than accepted at face value. Results:

**CONFIRMED — good practice, described imprecisely:** the claim that "a manifest was built once for ELA and reused unchanged for RGB and Fusion" is **not literally what the code does**. In reality, the RGB and Fusion scripts each **independently re-compute** their own split using the same seed (42) and the same procedural logic (sort → shuffle → take first 10%) — they do not load a shared, previously-saved manifest file. The practical outcome is similar (same procedure, same seed), but the correct thesis description is *"an identical split-construction procedure was independently reproduced across scripts,"* not *"a single manifest was built once and reused."* This is a wording-precision issue, not a methodology flaw.

**REFUTED — checked and found impossible by construction:** the claim that `file_to_source[filename]` risks a dictionary-key collision if, e.g., `fake001.jpg` exists in both CASIA1 and CASIA2. Checked against `generate_augmentations()` in the ELA generator: output filenames are built as `f"{source_dataset}_{cls}_{base_name}_aug{aug_idx}.png"` — the `source_dataset` is already embedded in the filename string itself, so two different source datasets can **never** produce colliding dictionary keys. No fix was needed.

**Independent corroboration of the above:** the RGB source-aware split's printed unique-original counts (10,281 train + 1,142 val = **11,423**) exactly match the independently-known total unique-source count from the ELA diagnostics (`{4: 11423}`). If any cross-source filename collision existed, this count would have come out lower (two different sources merging into one group). The exact match rules the collision claim out empirically as well as by code inspection.

**PARTIALLY UNVERIFIABLE / not found in the actual script shown:** the claim that a train↔test leakage-check block (`test_sources = {...}` / `intersection_test`) exists or is needed. The Fusion V2 script as actually run only contains a train↔val check, not a train↔test check. However: (a) `dataset_log.csv` **does** contain rows for both `train` and `test` splits — confirmed directly from the generator's `tasks` list, which explicitly includes both splits for all 4 source/class combinations, with `csv_writer.writerow()` called unconditionally per instance; and (b) train/test separation is structurally guaranteed further upstream anyway, since the very first generator script splits raw source images via `train_test_split(shuffle=True)` **before** any augmentation is generated — no runtime check can undo that. Adding an explicit train↔test check is cheap and harmless as a defensive measure, but it is not closing a real gap.

**ADOPTED AS GOOD PRACTICE, NOT YET APPLIED:** `torch.load(..., weights_only=True)` for the RGB/ELA checkpoint loads inside the Fusion script — current modern PyTorch recommendation to avoid unpickling arbitrary code. Low priority, safe to add whenever convenient, does not change any existing result.

**NOTED ASYMMETRY (low priority, not currently causing any known problem in the RGB/ELA baseline numbers, but caused a real downstream bug — see Section 13):** the plain RGB-only training script (Section 7) groups its source-aware split by `original_filename` alone, while the ELA (Section 9) and Fusion (Section 10) scripts group by the composite key `(source_dataset, original_filename)`. Since filenames provably don't collide across sources in this dataset (see the count-matching corroboration above), this makes no difference to the RGB baseline's own reported numbers. It *did*, however, cause the RGB-val and ELA-val prediction sets to silently diverge (same count, different actual files) when they were later joined for the Decision Fusion experiment — see Section 13 for the bug and its fix.

---

## 12. HYPOTHESIS TESTING — "DOES ELA CARRY INDEPENDENT SIGNAL?" (raised externally, refuted)

An external AI, reviewing the V1 vs. V2 Fusion results, proposed that ELA carries essentially no independent signal beyond what RGB already provides, built on five sub-hypotheses. The core supporting claim was that **"RGB ≈ Fusion ≈ 74%"** — i.e., that fusion showed no meaningful improvement over the RGB branch alone.

**This premise was checked against the actual logged numbers and found false.** RGB alone is **68%** accuracy / **0.67** macro F1 (Section 7) — not 74%. Fusion (74% / 0.74 macro F1) beats RGB by a real, non-trivial margin: **+6 accuracy points, +0.07 macro F1**. Three of the five proposed sub-hypotheses (that ELA is redundant, that ELA is "too weak" to matter, that "ResNet18 already extracted the ceiling from RGB alone") were built directly on this false premise and were set aside rather than pursued further, once the premise was checked and rejected.

**What survives independently of the false premise, and remains open (low urgency):**
- A JPEG-quantization-statistics confound (the model might partly be learning double-JPEG statistics rather than pure splice artifacts) — this is not a new concern, it is the same phenomenon already documented and closed in Section 4/5 (quantization resonance). Worth a cross-reference in the thesis, not a new investigation.
- The fusion MLP head possibly being undersized or overfitting quickly (its own loss curve shows train loss still dropping across epochs while val loss plateaus and rises early) — plausible on its own merits, cheap to test later (a smaller/more-regularized MLP head experiment), not yet run, not urgent.

**This was resolved with direct empirical evidence rather than further argument** — see Section 13 (Decision Fusion), which gives a clean, independent test of whether ELA adds value.

---

## 13. DECISION-LEVEL FUSION EXPERIMENT — TWO INVALID ATTEMPTS, THEN A CLEAN RESULT

**Purpose:** test whether a simple weighted average of RGB and ELA probabilities (no learned MLP, no shared training) also benefits from combining both branches. This is both (a) a direct, cheap test of the Section 12 hypothesis, and (b) a useful baseline to compare against the learned Feature-Fusion MLP.

### Attempt 1 — INVALID, do not cite
Alpha (101 values, linspace 0–1) and a fixed threshold of 0.5 were searched by scanning directly against the **test-set** predictions (`rgb_test_predictions.csv` / `ela_sourceaware_test_predictions.csv`). Result: **alpha=0.31, macro F1=0.7057**. Invalid because tuning any hyperparameter (here, alpha) directly against the test set is a form of information leakage, independent of how honest the underlying per-branch probabilities are. Kept only as a rough sanity-check upper bound, never as a reportable number.

### Attempt 2 — INVALID, do not cite (root-caused and explained, useful as a methodology cautionary note)
Regenerated val-set predictions and searched **jointly** over 101 alpha values × 91 threshold values (9,191 combinations) on the val set (n=4,568). Result: **val macro F1 = 0.9035** — higher than every other result in the entire project, which is itself a warning sign. Best threshold landed at an extreme value (0.12), another warning sign. Applied once to test: **macro F1 collapsed to 0.6569** — worse than either baseline branch alone (RGB 0.67, ELA 0.68). **Diagnosis:** classic multiple-comparisons overfitting — too fine a joint search grid relative to a small val set finds a combination that fits val-set noise rather than a real optimum.

**Also discovered and fixed along the way — a real split-mismatch bug:** the join between RGB-val and ELA-val predictions produced `NaN`s. **Root cause:** RGB's val split (Section 7) was built by grouping on `original_filename` alone, while ELA's val split (Section 9) groups on the composite key `(source_dataset, original_filename)`. Even with an identical seed (42), `shuffle()` applied to a list of plain strings vs. a list of tuples produces a **different resulting subset** — matching *count* (4,568) but not matching *content*. The two independently-generated val prediction CSVs therefore did not actually cover the same 4,568 images. **Fix:** regenerated `rgb_val_predictions.csv` using the same composite-key grouping logic as ELA, guaranteeing both val prediction sets cover identical files.

**Caveat, documented, not yet fully resolved, low priority:** the regenerated `rgb_val_predictions.csv` (composite-key split) is not literally identical to the split the RGB baseline model was originally trained/validated against (`original_filename`-only split, Section 7) — a small number of images may sit on different sides of train/val between the two versions. This does **not** affect the RGB baseline's own reported test metrics (test isolation is guaranteed at the raw-file level regardless of val-split construction). It only means the val set used to tune Decision Fusion's alpha/threshold is very slightly different from the val set the RGB model itself was checkpoint-selected against. Judged low-impact (same seed, same procedure, only the grouping key differs, and filenames provably don't collide across sources), but flagged for full rigor if ever revisited.

### Attempt 3 — VALID, this is the citable result
Two-stage **sequential** search instead of a joint grid, specifically to avoid the Attempt-2 overfitting mode:
1. **Stage 1:** search alpha only (101 values), threshold fixed at 0.5. Best: **alpha = 0.83** (i.e., weighting RGB more heavily than ELA), val macro F1 (at threshold 0.5) = 0.7678.
2. **Stage 2:** with alpha fixed at 0.83, search the threshold via `sklearn.metrics.roc_curve`'s candidate thresholds on val, pick the argmax macro F1. Best threshold = **0.2276**, val macro F1 = 0.8095.
3. Applied **once** to the test set (never touched during either search stage): **macro F1 = 0.7315, AUC = 0.7892**.

**Val→test gap check:** 0.8095 → 0.7315 (Δ=0.078) is a normal, expected order of magnitude for this project's val size (n=4,568) — a healthy contrast against Attempt 2's Δ=0.25 gap, which was the actual tell that something was wrong there.

**Interpretation:** alpha=0.83 (RGB weighted more heavily than ELA) is directionally consistent with RGB being the very slightly stronger single branch in this pairing (68% vs. 69% is close, but recall ELA edges RGB marginally as a standalone model — the alpha value reflects the fusion optimization surface, not simply "which branch is better alone"). Critically, ELA's weight is **not driven to zero** — if ELA were truly redundant with RGB, decision-level fusion would gain nothing over RGB alone, and alpha would optimize toward 1.0. It doesn't; see Section 14 for why this matters.

---

## 14. FOUR-WAY COMPARISON (data now exists; NOT yet consolidated into one script/CSV — remaining task)

| Model | Test Accuracy | Test Macro F1 | Test AUC | Threshold policy |
|---|---|---|---|---|
| RGB-only (`model_resnet_rgb_LATEST.pth`) | 68% | 0.67 | not computed/reported | fixed 0.5 |
| ELA-only, source-aware (`model_resnet_ela_sourceaware_v2.pth`) | 69% | 0.68 | not computed/reported | fixed 0.5 |
| Decision Fusion (α=0.83, two-stage val search) | not computed as accuracy | **0.7315** | **0.7892** | val-based (roc_curve, macro F1) |
| Feature Fusion MLP V2 (`model_fusion_sourceaware_v2.pth`) | **74%** | **0.74** | **0.802** | val-based (roc_curve, macro F1) |

**Clear, monotonic ordering:** RGB ≈ ELA < Decision Fusion < Feature Fusion.

**Why this matters:** this directly and empirically refutes the Section 12 hypothesis that ELA carries no independent signal. A pure linear probability blend — no learned interaction, no shared training, just a weighted average — already gains ~5–6 points of macro F1 over either single branch. This would be structurally impossible if ELA and RGB encoded fully redundant information; averaging two identical signals cannot manufacture a gain that neither signal has alone.

**The gap between Decision Fusion (0.7315) and Feature Fusion (0.74)** is small (~1 point / 0.013 AUC) — suggesting the learned MLP head extracts a modest additional amount of value over simple weighted averaging, but **most** of the fusion benefit is already captured by the cheap linear blend. This is a good, citable nuance for the thesis discussion section: "decision-level fusion already captures most of the benefit; feature-level fusion adds a smaller additional increment, justifying the added architectural complexity only modestly."

**Remaining gap in the comparison, not yet closed:** RGB and ELA baselines above use a fixed threshold of 0.5, while both fusion variants use val-based macro-F1-optimized thresholds. This mismatch was flagged early in this project (by the very first external review, well before this phase) and is still not fully resolved. To make the four-way table fully apples-to-apples, RGB and ELA should also get a macro-F1-optimized threshold derived from their own val sets (same `roc_curve` pattern already used everywhere else in this project). Cheap, not yet executed.

**This table substantively answers the long-standing "does fusion help" question** (v5 Section 9, item 1) even though the originally-drafted dedicated three-way comparison script was never formally run as a single script — the same question ended up answered piecemeal through the ELA retrain and Decision Fusion work instead. **Recommended remaining task:** consolidate into one clean script + saved CSV for the thesis appendix, fixing the threshold-policy mismatch noted above at the same time.

---

## 15. EMBEDDING-SCALE SANITY CHECK (cheap diagnostic, ruled out)

**Hypothesis tested:** maybe the fusion MLP favors one branch simply because its raw embedding magnitude is larger, independent of actual information content — which would be a fixable normalization issue rather than a real data-content finding.

**Measurement** (on the saved test-set `.npy` embeddings): RGB L2-norm mean/std = **27.51 / 6.13**; ELA L2-norm mean/std = **24.45 / 4.85**. Ratio ≈ **1.12×**.

**Conclusion:** the ratio is small — this kind of effect would plausibly need to be on the order of 2× or more to meaningfully explain branch-dominance behavior in a concatenation+MLP setup. **Hypothesis rejected, no action taken.** L2-normalizing embeddings before concatenation was considered but not pursued, since the prior evidence for a problem was weak and the fix would require re-exporting embeddings and retraining the MLP head for uncertain benefit.

---

## 16. IMMEDIATE NEXT ACTIONS (updated, reprioritized — replaces v5 Sections 9–10)

**HIGH PRIORITY:**
1. **Cross-dataset holdout** (Columbia and/or IMD2020) — still the single most important missing piece for the generalization claim, untouched since v4. Needs an explicit, **pre-registered** inference protocol decided *before* opening any external-dataset results, to avoid the cross-dataset evaluation turning into a second, hidden hyperparameter search:
   - What primary JPEG-quality handling applies to external RGB images (raw image, or simulated at some fixed `q_primary`)?
   - What `q_ela` / ELA construction settings apply (fixed, or the same randomized scheme — external images won't come with the synthetic multi-quality metadata CASIA does)?
   - Which threshold is reused — only the one already selected on CASIA validation, nothing tuned on the external set.
2. **Frontend hardcoded placeholder sub-metrics / fake EXIF fields** in the "AEGIS // VISION" prototype — unresolved since the original build. Cheap to fix, meaningful defense-day credibility risk if left to the last week.

**MEDIUM PRIORITY:**
3. Consolidate the four-way comparison (Section 14) into a single script + saved CSV, and close the RGB/ELA threshold-policy mismatch (macro-F1 threshold from their own val sets, replacing the current fixed 0.5) so all four numbers are computed under an identical protocol.
4. Server-side `INFERENCE_ELA_QUALITY` final pick (Q=85 average-best vs. Q=95 stability-best, Section 4) — needs a decision before any retrained/fused model reaches the live demo; naturally pairs with item 1, since external-dataset images won't arrive with the synthetic multi-quality scheme either.
5. Thesis paragraphs — material is now ready for three strong, self-contained methodology narratives:
   - (a) The RGB/ELA pairing bug and fix (carried from v5, Section 6).
   - (b) The ELA source-aware retrain as a proactive integrity check with a reassuring null result (Section 9).
   - (c) The false "ELA carries no signal" hypothesis, checked against real numbers, found false, and independently resolved via Decision Fusion (Sections 12–14) — hypothesis raised → premise checked → premise rejected → resolved with a clean, cheap, additional experiment. This is a genuinely strong "honestly-obtained, calibrated result" narrative for a committee that explicitly values that over polish.

**LOW PRIORITY / OPTIONAL:**
6. GradCAM placement in the `robustness/` / `model_evaluation/` / `data_integrity/` test-folder structure — still open, non-blocking.
7. `torch.load(..., weights_only=True)` — modern-practice cleanup, cosmetic, no urgency (Section 11).
8. The RGB-vs-ELA split-grouping-key asymmetry (`original_filename` vs. composite key, Section 11) — documented, no evidence it affects any *reported* result, but flagged for full rigor; would only matter if Decision Fusion tuning is revisited at higher precision.
9. **Joint fine-tuning of `layer4`** in both backbones ("Jointly Fine-Tuned Late Fusion") — a legitimate follow-up now that Decision Fusion has confirmed ELA carries real, usable independent signal. Technical requirements if pursued: remove the `torch.no_grad()` wrapper around both branches inside `forward()` (currently blocks all gradient flow to the backbones even if `requires_grad=True` is set); keep BatchNorm forced to `.eval()` via the existing `train()` override pattern; use a much smaller LR (~1e-5) for the newly-unfrozen `layer4` parameters vs. the MLP head's LR (~1e-4); add gradient clipping; save as a **separate** checkpoint (do not overwrite `model_fusion_sourceaware_v2.pth`) since this becomes a structurally different model, not a retrain of the same one. **Not started.** Per guardrails (diagnose-before-architect), should come *after* the cross-dataset holdout, not before — the potential gain here (~1–3 F1 points, unconfirmed) is smaller than what's structurally at stake in the still-untested generalization claim.
10. Smaller / more-regularized MLP head experiment (cheap, addresses Section 12's surviving Hypothesis 3 about the head possibly being undersized) — not yet run, low urgency.
11. PCA/t-SNE or CKA/correlation analysis on the saved embeddings (`.npy` files already exported for both V1 and V2 Fusion runs) — cheap, would make a nice supporting figure, not yet run.

---

## 17. KEY REUSABLE ARTIFACTS (updated inventory — supersedes v5 Section 11)

**Carried forward unchanged from v4/v5:**
- `build_source_aware_val_split()` pattern — verified leak-free; this phase it was also independently re-implemented for the ELA and Fusion training scripts using a composite key (see Section 11 for the "reproduced procedure" vs. "reused manifest" wording distinction).
- `sweep_manifest_n1000.csv`, `sweep_metrics_results.csv` — locked ELA quantization-resonance sweep data.
- `RGB_v2` generator script — reads `q_primary`/`split`/`class` directly from the ELA log; zero-mismatch verified.
- `CASIA_Unified_Randomized_RGB_v2/` + its `dataset_log.csv`.
- `model_resnet_rgb_LATEST.pth` — RGB baseline, 68% acc / 0.67 macro F1, unchanged this phase.

**New this phase:**
- `model_resnet_ela_sourceaware_v2.pth` — retrained ELA branch, source-aware composite-key split, 69% acc / 0.68 macro F1 (Section 9).
- `ela_sourceaware_split_v2.csv` — split manifest (`source_dataset, original_filename, subset`).
- `ela_sourceaware_test_predictions.csv` — ELA test-set predictions (`filename, true_label, pred_label, prob_fake`).
- `ela_val_predictions.csv` — ELA val-set predictions, same composite-key split used at training time.
- `model_fusion_sourceaware_v2.pth` — Fusion retrained on the new ELA branch + `label_smoothing=0.05` added to the fusion-head criterion (Section 10). 74% acc / 0.74 macro F1 / AUC 0.802, threshold 0.2893.
- `fusion_v2_test_metadata.csv` + re-exported embeddings in a `fusion_v2/` output folder (`test_embeddings_rgb_512d.npy`, `test_embeddings_ela_512d.npy`, `test_embeddings_fused_1024d.npy`, `test_labels.npy`).
- `rgb_val_predictions.csv` — RGB val-set predictions, **regenerated using the composite-key split** for Decision Fusion compatibility with ELA's val split (see Section 13 caveat about the small asymmetry vs. RGB's own original training split).
- Decision Fusion tuning script — the **corrected, two-stage version** (alpha-only search at threshold=0.5, then threshold search via `roc_curve` at fixed alpha). **Earlier one-shot (test-tuned) and joint-grid (val-overfit) versions must not be reused or cited** — see Section 13, Attempts 1–2.
- Four-way comparison numbers (Section 14) — exist as separate run outputs scattered across this project's history; **not yet consolidated into one script/CSV** (flagged as Medium Priority action item 3).

---

## 18. GUARDRAILS (unchanged, still binding)

- Topic is fixed. Do not propose changing it.
- Cap total additional datasets at 2–3 beyond CASIA1+CASIA2 (Columbia and/or IMD2020 preferred).
- Do not add a third detection algorithm (e.g., copy-move via keypoint matching) — flagged by the student as scope creep to avoid.
- Prefer diagnosing before architecting — validated repeatedly across every phase so far, including this one (the ELA retrain, the split-key mismatch bug in Decision Fusion, and the false-premise hypothesis were all caught by checking/diagnosing before further training or further argument, not after).
- Keep language calibrated for a mathematically literate committee: state sample sizes, avoid "proven"/"cured"-style claims where "moderate, reproducible improvement" is the honest description.
- Avoid re-litigating closed conclusions via further AI-opinion-gathering; when an external AI's claim can be checked against actual code or actual numbers, check it directly rather than collecting a third opinion (Section 1, Section 11, Section 12 are all examples of this working as intended this phase).
- Execute the next action (Section 16) directly — cross-dataset holdout is the standing highest-priority item and should not slip further.
