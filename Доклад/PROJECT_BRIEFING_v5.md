# PROJECT BRIEFING v5: Image Manipulation Detector — Master's Thesis
### (Context transfer — supersedes `PROJECT_BRIEFING_v4.md`. Read fully before advising.)

This document consolidates everything resolved since v4: two data/code integrity bugs found and fixed in the ELA→RGB pivot, a retrained RGB baseline, and the first fully-verified Late Fusion result. Do not propose changing the topic. Do not propose architecture changes or new datasets/signals without checking Section 12 (guardrails) first.

---

## 1. WHO I AM

Master's student in Computer Science. Committee: applied mathematicians, microcontroller/acoustics/ballistics specialists — not deep-ML experts. Narrative clarity and honestly-obtained, calibrated results matter more than implementation minutiae.

**Psychological note, updated:** the earlier pattern (gathering sequential opinions before acting) showed a live test case this phase — a second AI flagged a possible RGB/ELA dataset pairing issue, and instead of collecting a third opinion, the finding was independently cross-checked, confirmed, fixed, and re-run. That's the target behavior going forward: verify once, act, move on. Two more bugs (BatchNorm freeze, fixed-quality on-the-fly ELA) were caught and closed the same way. Keep reinforcing this pattern; no signs of regression to serial opinion-shopping.

---

## 2. TOPIC (FIXED — DO NOT SUGGEST CHANGING)

Image manipulation detector with localization and explainability. Three components: (1) ELA module — hand-implemented, non-ML; (2) CNN classifier; (3) GradCAM explainability. **Current phase:** Late Fusion (ELA + RGB) is now trained on a fully verified pipeline with a clean, trustworthy result. Next: a controlled three-way comparison (ELA vs. RGB vs. Fusion), then cross-dataset holdout.

---

## 3. FOUNDATIONAL WORK (COMPLETE — unchanged from v4)

- Hand-implemented ELA (`ela.py`); strong signal on differing compression histories, near-zero signal on uniformly-resaved composites (the "bird-in-garden" case).
- Early CASIA1-only baseline: raw RGB CNN (~51%, no signal) vs. ELA+ResNet18 full fine-tune (86% acc, F1 0.85). Note: the ~51% RGB number is now understood to reflect that early setup, not RGB's ceiling — see Section 7.
- GradCAM confirmed the network attends to splice boundaries, not object interior.
- FastAPI + frontend prototype ("AEGIS // VISION") exists. **Still carries hardcoded placeholder sub-metrics and fake EXIF fields — unresolved, a defense risk if raised late.**

---

## 4. THE QUANTIZATION RESONANCE FINDING (established, unchanged)

V1 (fixed `q_ela=75`) showed a sharp confidence collapse exactly where `Q_primary` matched the hardcoded ELA reference quality — a property of the JPEG-Ghosts-style delta math, not data narrowness. The x4/Kaggle model (randomized `q_ela` during training) raised the resonance floor substantially (worst-case accuracy ~63% vs. V1's near-chance collapse) but did not eliminate the resonance dip itself. Full sweep results archived in `sweep_metrics_results.csv`; Q=85 wins on average, Q=95 wins on stability/floor — stated as an explicit trade-off in the thesis, not a single "best" answer.

---

## 5. ELA PHASE — CLOSED (unchanged)

Sufficient evidence at appropriate scale supports the resonance limitation, the augmentation-driven improvement, and the server-quality trade-off. This conclusion is not being re-opened.

---

## 6. RGB DATASET PAIRING BUG — FOUND AND FIXED

**The bug:** the original RGB-branch generator and the ELA generator were meant to apply the *same* primary JPEG quality (`q_primary`) per augmented instance, so a filename-matched RGB/ELA pair would represent the same compression scenario viewed two ways. They didn't. Root cause was two compounding issues, both in the RGB generator:

1. It re-sampled a fresh `q_primary` via `random.sample()` instead of reusing the value already fixed by the ELA generation run.
2. It preserved raw `os.listdir()` file order when filtering by the split map, while the ELA generator's `train_test_split(shuffle=True)` had reshuffled that order — so even the file *processing sequence* differed between the two scripts, on top of the RNG-state drift from unequal numbers of `random.sample()` calls per file.

**Severity, empirically confirmed:** a simulation reproducing both scripts' exact logic showed a 10.8% match rate on `q_primary` between filename-paired files — statistically indistinguishable from the ~9.1% expected under full decorrelation (11 possible quality values). Effectively every pair was scrambled, not just a "drifts after image 1" issue.

**Blast radius:** did NOT affect the ELA-only model, the train/test split integrity, or the RGB-only model's validity *as a standalone result* (RGB-only training never reads ELA data). DID invalidate every Late Fusion training/eval run done before the fix, since that's the only place cross-branch pairing matters.

**Fix:** rebuilt the RGB generator (`RGB_v2`) to stop sampling entirely — it reads `filename`, `split`, `class`, and `q_primary` directly from the ELA log's CSV and replays them exactly, guaranteeing pairing by construction rather than by coincidence.

**Verification:** row-count, `q_primary`, and `split` diffed to zero mismatches between the two logs, confirmed directly (not just simulated).

---

## 7. RGB-ONLY BASELINE — RETRAINED ON v2, RESULT LOCKED IN

Full fine-tune ResNet18 (ImageNet-pretrained), same recipe/hyperparameters as the ELA branch for a fair comparison, source-aware split with leak check passing. Trained on the corrected `RGB_v2` dataset; checkpoint provenance confirmed by the student.

- Early stopped at epoch 5, best checkpoint = epoch 1 (val loss 0.6354).
- Test set (11,432 images), threshold = default 0.5:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| REAL | 0.71 | 0.76 | 0.73 |
| FAKE | 0.64 | 0.57 | 0.60 |

Accuracy 68%, macro F1 0.67. Confirms raw RGB carries real forensic signal with a proper backbone — the early ~51% "no signal" finding reflected that experiment's setup, not a ceiling on RGB itself. This result and checkpoint are the ones referenced by Fusion (Section 8).

---

## 8. LATE FUSION — TWO MORE BUGS CLOSED, FIRST CLEAN RESULT OBTAINED

Two additional issues surfaced and were resolved before trusting any Fusion number:

**BatchNorm freeze bug:** `model.train()` is recursive — it was flipping the frozen `rgb_branch`/`ela_branch` back into train-mode BatchNorm every epoch (batch statistics instead of running statistics), even though their weights were frozen via `requires_grad=False`. This created a train/eval embedding mismatch for the "frozen" backbones. Fixed by overriding `train()` on the fusion model to force both branches to `.eval()` unconditionally, regardless of the outer training-mode flag.

**Fixed-quality on-the-fly ELA (found, then avoided by design change rather than patched):** an alternative dataset design that computed ELA on-the-fly from the RGB image (rather than loading a pre-generated ELA file) used a hardcoded `ela_quality=90` for every sample — silently reintroducing the exact fixed-quality resonance setup the project had already moved past via x4 randomized-quality training. Reverted to the offline paired-dataset design instead, which is now safe to use as-is because `RGB_v2` guarantees correct RGB/ELA pairing by construction (Section 6).

**Threshold policy corrected:** switched the validation threshold search from FAKE-only F1 to macro F1. The earlier FAKE-only version had picked threshold 0.24 and produced a lopsided 0.61/0.90 REAL/FAKE recall split — a methodology artifact of the optimization target, not a finding about the model.

**Clean result** (correct pairing + BN fix + confirmed `RGB_v2` checkpoint + macro-F1 threshold):

- Train 41,124 / Val 4,568 / Test 11,432 (source-aware split, leak-free).
- Early stopped at epoch 7, best checkpoint = epoch 3 (val loss 0.6321).
- Threshold (macro F1) = 0.261.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| REAL | 0.87 | 0.67 | 0.75 |
| FAKE | 0.65 | 0.86 | 0.74 |

Accuracy 75%, macro F1 0.75, **AUC 0.806**.

**Status:** first Fusion result built on a fully verified pipeline — this is the one to carry toward the thesis, pending the controlled comparison in Section 9. Two earlier Fusion runs (73%/AUC 0.796 pre-pairing-fix; anything pre-BN-fix) are superseded and should not be reported as thesis results, though they may be worth a one-line "before/after the integrity fix" footnote if useful for methodology narrative.

---

## 9. IMMEDIATE NEXT ACTIONS

1. **[NOT YET RUN]** Controlled three-way comparison — ELA-only vs. RGB-only vs. Fusion, identical test set, identical val-based macro-F1 threshold protocol for all three. Script already drafted (pure evaluation on existing checkpoints, no retraining). This converts "75% Fusion vs. 68% RGB vs. ~68–70% ELA-sweep-average" from an eyeballed comparison into a rigorous, citable one — needed before claiming fusion outperforms either branch alone.
2. **Then:** cross-dataset holdout (Columbia/IMD2020) — see Section 10, item 2.

---

## 10. OPEN ITEMS (prioritized)

1. **[HIGH PRIORITY]** Run the drafted three-way comparison script — fast, no training required, directly answers "does fusion help."
2. **[HIGH PRIORITY, LONGSTANDING]** No cross-dataset holdout has been run yet. Still the single most important missing piece for the generalization claim. Also forces the open server-Q decision (item 4) as a side effect, since cross-dataset images won't come with the synthetic multi-quality scheme.
3. Frontend hardcoded placeholder sub-metrics / fake EXIF — unresolved, not blocking ML work, defense risk if left to the last week.
4. Server-side `INFERENCE_ELA_QUALITY` final pick (Q=85 vs. Q=95) — needs an explicit decision before any retrained/fused model reaches the live demo.
5. GradCAM not represented in the `robustness/`, `model_evaluation/`, `data_integrity/` test-folder structure — decide if it belongs in the universal eval package or stays prototype-only.
6. Thesis draft paragraph needs revision (carried from v4's note on the old `Brief2_diplom.txt` fragment), **plus** a new short paragraph documenting the RGB/ELA pairing bug and fix — this is a methodology-rigor point worth stating plainly to the committee, not something to bury.

---

## 11. KEY REUSABLE ARTIFACTS (updated inventory)

- `build_source_aware_val_split()` — integrated, verified leak-free (unchanged from v4).
- `sweep_manifest_n1000.csv`, `sweep_metrics_results.csv` — locked ELA sweep data (unchanged from v4).
- **`RGB_v2` generator script** — corrected version; reads `q_primary`/`split`/`class` directly from the ELA log, no independent sampling.
- **`CASIA_Unified_Randomized_RGB_v2/` + its `dataset_log.csv`** — verified zero-mismatch against the ELA log on row count, `q_primary`, and `split`.
- **`model_resnet_rgb_LATEST.pth`** — retrained on `RGB_v2`, provenance confirmed.
- **`model_fusion_LATEST.pth`** — trained on corrected data, BN-safe architecture, macro-F1 threshold (0.261).
- `fusion_test_metadata.csv` + exported embeddings (RGB 512d, ELA 512d, fused 1024d) — from the clean Fusion run; ready for t-SNE/PCA or branch-ablation follow-up once the three-way comparison and cross-dataset work are done.
- **Three-way comparison eval script** — drafted, not yet executed (Section 9, item 1).

---

## 12. GUARDRAILS (unchanged, still binding)

- Topic is fixed. Do not propose changing it.
- Cap total additional datasets at 2–3 beyond CASIA1+CASIA2 (Columbia and/or IMD2020 preferred).
- Do not add a third detection algorithm (e.g., copy-move via keypoint matching) — flagged by the student as scope creep to avoid.
- Prefer diagnosing before architecting — validated repeatedly across every phase so far, including this one (two of three bugs this phase were caught by diagnosis before further training, not after).
- Keep language calibrated for a mathematically literate committee: state sample sizes, avoid "proven"/"cured"-style claims where "moderate, reproducible improvement" is the honest description.
- Avoid re-litigating closed conclusions via further AI-opinion-gathering; execute the next action (Section 9) directly.
