# PROJECT BRIEFING v4: Image Manipulation Detector — Master's Thesis
### (Context transfer — supersedes `Brief_diplom.txt` / "v2" and `Brief2_diplom.txt`. Read fully before advising.)

This document consolidates everything resolved since the last full briefing snapshot (v2/v3), plus the state of the ELA→RGB pivot. Do not propose changing the topic. Do not propose architecture changes or new datasets/signals without checking Section 8 (guardrails) first. Note on sources: `Brief_diplom.txt` was an earlier snapshot (ends before the split-fix was integrated); `Brief2_diplom.txt` was not a briefing — it was two pasted fragments (a draft thesis paragraph + a copied chat message) and should not be used as a standalone reference going forward.

---

## 1. WHO I AM

Master's student in Computer Science. Committee: applied mathematicians, microcontroller/acoustics/ballistics specialists — not deep-ML experts. Narrative clarity and honestly-obtained, calibrated results matter more than implementation minutiae.

**Psychological note, carried forward:** documented pattern of gathering sequential opinions from multiple AI assistants before acting on a decision. Recent behavior shows real improvement — decisions are now being executed (split-fix integrated, large-scale sweep run) rather than endlessly re-opinion-collected. Keep encouraging convergence-then-execution over further comparison shopping, especially now that the ELA phase is closing and RGB work is starting fresh (a natural trigger point for the pattern to recur).

---

## 2. TOPIC (FIXED — DO NOT SUGGEST CHANGING)

Image manipulation detector with localization and explainability. Three components: (1) ELA module — hand-implemented, non-ML; (2) CNN classifier; (3) GradCAM explainability. Current phase: pivoting from ELA-only to a two-branch (ELA + RGB) Late Fusion architecture, motivated by a now well-evidenced structural ceiling in ELA alone (see Section 5).

---

## 3. FOUNDATIONAL WORK (COMPLETE — unchanged from earlier briefings)

- Hand-implemented ELA (`ela.py`); found strong signal on differing compression histories, near-zero signal when a composite is uniformly resaved (the "bird-in-garden" case).
- CASIA1-only baseline experiments: raw RGB CNN (~51%, no signal) vs. ELA+ResNet18 full fine-tune (86% acc, F1 0.85 — best result at the time).
- GradCAM confirmed the network attends to splice boundaries (JPEG block-grid mismatch), not object interior.
- FastAPI + frontend prototype ("AEGIS // VISION") exists. **Still carries hardcoded placeholder sub-metrics and fake EXIF fields — unresolved, a defense risk if raised late.**

---

## 4. THE QUANTIZATION RESONANCE FINDING (established, Phase 2)

Diagnostic sweeps (single image → 3-image → 100+100 "ultimate test") on **V1** (fixed training-time `q_ela=75`) found a sharp confidence spike/collapse exactly where `Q_primary` matches the hardcoded ELA reference quality (75). At that resonance point, V1's accuracy and FAKE recall dropped toward chance level across the sweep (Section 5.4/14.3 in earlier briefings: recall collapsed to ~32% at the resonance point).

**Root cause:** the ELA signal is fundamentally a delta between two JPEG quantization passes. When primary and inference-time quality coincide, the delta collapses regardless of content — this is a property of the math (JPEG-Ghosts-style artifact), not a data-narrowness issue.

---

## 5. RESOLVED THIS PHASE: SPLIT INTEGRITY + LARGE-SCALE ELA SWEEP

### 5.1 Source-aware train/val split — INTEGRATED AND VERIFIED (previously open item, now closed)
`build_source_aware_val_split()` is integrated. Leakage check re-run with an explicit intersection test:
```python
intersection = train_originals & val_originals_check
```
Result: `len(intersection) == 0` — confirmed no leakage between train and val at the source-image level. This resolves the standing concern that val-based checkpoint/threshold selection (used for the x4/Kaggle model, threshold 0.4327) might have been distorted by near-duplicate leakage.

### 5.2 Manifest-locked N=1000 sweep — the "final chord" ELA test
Script: `robustness/eval_server_inference_sweep.py` (evolved from the earlier N=3 and N=200 diagnostics). Key methodological improvements over earlier sweeps:
- **Manifest persistence** (`sweep_manifest_n1000.csv`): fixes the earlier problem (Section 14.4 in prior briefing) where two comparison runs could not be proven to use identical image samples.
- Tested on the **x4/Kaggle model** (randomized `q_ela` during training) — not V1.
- Swept `SERVER_INFERENCE_SETTINGS = [75, 85, 95]` × `TEST_QUALITIES` (9 primary-Q values), N=1000 (500 REAL + 500 FAKE from the test split).

### 5.3 Results — precise, computed from `sweep_metrics_results.csv`

| Server_Q | Mean Accuracy | Mean Recall | Mean MCC | StdDev Acc | StdDev Recall | StdDev MCC | Min Recall (floor) |
|---|---|---|---|---|---|---|---|
| 75 | 67.9% | 70.8% | 0.364 | 4.57 | 9.44 | 0.085 | 52.0% |
| 85 | **70.6%** | **73.1%** | **0.418** | 4.31 | 11.64 | 0.084 | 53.6% |
| 95 | 69.0% | 66.9% | 0.381 | **3.82** | **4.14** | **0.077** | **61.2%** |
(In can be uncertein, more detail in .csv file)

**Conclusions (calibrated, not overclaimed):**
1. **Resonance persists but is no longer catastrophic.** Every server_q setting shows a local dip in MCC/Accuracy exactly at its own resonance point (`Q_primary == server_q`), confirming resonance is a physical property of JPEG quantization math, not something eliminated by training-time augmentation. However, even the worst dip (Accuracy 62.8% at server=75/primary=75) sits well above chance — a large improvement over V1's near-chance collapse. x4-augmentation raised the *floor*; it did not remove the resonance mechanism.
2. **Resonance shows mainly as false positives, not missed fakes.** FAKE Recall dips do not consistently align with resonance points the way Accuracy/MCC do — consistent with the earlier "ranking inversion" finding that resonance primarily inflates false positives on REAL images rather than blinding the model to real fakes.
3. **Server Q trade-off (open, non-blocking product decision):** Q=85 wins on average performance; Q=95 wins decisively on stability and worst-case floor (recall never drops below 61% vs. 52–54% for the others). This is a genuine trade-off to state explicitly in the thesis, not a single "best" answer — framing should be "best average" vs. "most predictable under unknown input quality."
4. **Self-correction, for calibration hygiene:** an earlier visual read of the MCC plot suggested Q=95 was *less* stable on MCC specifically. Precise computation showed the opposite — Q=95 has the lowest StdDev on MCC too (0.077 vs. 0.084–0.085). Eyeballing plots is not a substitute for computing exact statistics before writing thesis claims — worth remembering as a general lesson, not just for this result.

### 5.4 Suggested thesis conclusion language (replaces "bingo, augmentation cured it")
> x4-augmentation with randomized `q_ela` substantially raised baseline performance and eliminated the catastrophic collapse-to-chance observed in V1. However, quantization resonance persists as a moderate, reproducible degradation exactly where primary and server-side compression quality coincide, regardless of the server setting chosen — confirming resonance as an inherent property of JPEG quantization mathematics rather than an artifact of training-data narrowness. This reinforces the ELA semantic-blindness ceiling and directly motivates the RGB Late Fusion architecture.

**Note on `Brief2_diplom.txt`'s draft paragraph:** it describes an "N=200" test with near-total recall collapse — this matches the older V1-era 100+100 diagnostic, not the N=1000 x4 sweep above. If reused in the thesis, **rewrite it to reference the N=1000 result and the calibrated conclusion above**, or clearly scope it as describing V1's earlier, more severe failure mode for historical/comparative purposes. Do not present both findings under one unlabeled "N=200" claim.

---

## 6. ELA PHASE — OFFICIALLY CLOSED

Sufficient evidence at appropriate scale (not anecdotal) now supports: (a) a genuine, physically-grounded resonance limitation, (b) a real augmentation-driven improvement over V1, (c) an evidence-based server-quality trade-off. Proceed to RGB.

---

## 7. IMMEDIATE NEXT ACTION: RGB-ONLY BASELINE

- Reuse the same source-aware split (`build_source_aware_val_split()`), same CSV log, same seed.
- Reuse the same evaluation harness pattern (classification report, ROC/PR-AUC, threshold search, confusion matrix) — do not rewrite from scratch.
- Train ResNet18 full fine-tune on raw RGB crops (no ELA transform) as the control group.
- Compare against the ELA branch on an identical, manifest-locked test set.
- **Then:** Late Fusion (ELA branch + RGB branch → concatenation → MLP head).
- **Then:** cross-dataset holdout (Section 8, item 1) for whichever architecture(s) survive to that point.

---

## 8. OPEN ITEMS (prioritized, carried forward)

1. **[HIGH PRIORITY, LONGSTANDING]** No cross-dataset holdout (Columbia / IMD2020) has been run yet. Still the single most important missing piece for the thesis's generalization claim. Not blocking RGB baseline start, but should not slip to the last month.
2. Frontend hardcoded placeholder sub-metrics / fake EXIF — unresolved since original build. Not blocking ML work; defense risk if left to the last week.
3. Server-side `INFERENCE_ELA_QUALITY` final pick (Q=85 vs. Q=95, see Section 5.3) — needs an explicit decision before any retrained/fused model reaches the live demo. Not blocking RGB work.
4. GradCAM is not represented in the new `robustness/` `model_evaluation/` `data_integrity/` test-folder structure — decide whether it belongs in the universal evaluation package or stays prototype-only.
5. Thesis draft paragraph (from `Brief2_diplom.txt`) needs revision per Section 5.4 note above before being finalized into the thesis text.

---

## 9. KEY REUSABLE ARTIFACTS (updated inventory)

- `build_source_aware_val_split()` — **now integrated and verified leak-free**, no longer pending.
- `sweep_manifest_n1000.csv` — locked N=1000 test sample (500/500), reusable for any future ELA-branch comparison.
- `sweep_metrics_results.csv` — full per-(server_q, primary_q) metrics, source for Section 5.3 table.
- Test folder structure: `ela_methods/`, `model_evaluation/`, `robustness/`, `data_integrity/` — grouped by test purpose, not creation order.
- FastAPI + frontend prototype ("AEGIS // VISION") — functional, placeholder-metrics issue still open (Section 8, item 2), not yet updated with a retrained/fused model or a decided server ELA-quality strategy.

---

## 10. GUARDRAILS (unchanged, still binding)

- Topic is fixed. Do not propose changing it.
- Cap total additional datasets at 2–3 beyond CASIA1+CASIA2 (Columbia and/or IMD2020 preferred).
- Do not add a third detection algorithm (e.g., copy-move via keypoint matching) — flagged by the student as scope creep to avoid.
- Prefer diagnosing before architecting — validated repeatedly across every phase so far.
- Keep language calibrated for a mathematically literate committee: state sample sizes, avoid "proven"/"cured"-style claims where "moderate, reproducible improvement" is the honest description.
- Avoid re-litigating the now-closed ELA conclusion via further AI-opinion-gathering; execute the RGB baseline next.
