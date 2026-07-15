# RUNLOG

## Run 1: Baseline — Naive 50/50 Blend
- **Script**: `blend.py --reference_dir ../reference`
- **Strategy**: Score all 54 stock voices against target. Blend top-2 (af_sarah + bf_emma) 50/50.
- **Scores**: Single-sentence (fox text) similarity = **0.7963**. Three-sentence avg = **0.5848**.
- **Heard**: Generic American female voice. Clear speech, correct words. Accent and timbre are completely different from the target speaker (Indian English, conversational support style).
- **Changed**: Nothing — this is the baseline to beat.

---

## Run 2: Convex Blend Weight Search (Top 5 Stock Voices)
- **Script**: `search.py --reference_dir ../reference --iters_blend 100 --iters_pert 0`
- **Strategy**: Random walk on the weight simplex over the top 5 stock voices. Fitness = avg cosine similarity on 3 reference-transcript sentences.
- **Best weights found**: af_sarah 2.4%, af_sky 24.2%, bf_lily 48.7%, af_nova 24.7%
- **Scores**: Three-sentence avg = **0.6034** (vs baseline 0.5848, improvement **+0.0186**).
- **Heard**: Richer, more natural voice. Blending multiple British and American voices softened the accent slightly. Still not Indian English but noticeably different timbre from the naive blend.
- **Changed**: Expanded to top 5 voices, replaced single-sentence eval with 3-sentence avg to prevent overfitting.

---

## Run 3: Broadcasted Perturbation with Soft L2 Penalty (Failed)
- **Script**: `search.py` with `reg_coef=0.05`, perturbation clamped to `[-0.15, 0.15]`
- **Strategy**: Add a shared 256-dim vector to all rows of the blend voice. Penalize L2 norm to prevent degradation.
- **Scores**: Three-sentence avg = **0.6034** — identical to Run 2, no improvement.
- **Heard**: No change from Run 2. Phase 2 produced zero accepted steps.
- **What went wrong**: The soft penalty immediately deducted ~0.016 from any non-zero step. In 256-dim space, a random step rarely improves cosine similarity by that much in one shot. All proposals were rejected. Regularization was too aggressive.
- **Changed**: Switched from soft L2 penalty to hard clamping.

---

## Run 4: Broadcasted Perturbation with Hard Clamping (Successful)
- **Script**: `search.py --reference_dir ../reference --iters_blend 60 --iters_pert 80 --step 0.02`
- **Strategy**: Remove soft penalty. Clamp perturbation elements to `[-0.05, 0.05]` (1/3 of stock voice std dev). Evaluate on 3 sentences.
- **Scores**: Three-sentence avg = **0.6139**. Phase 2 accepted steps and improved beyond blend-only baseline.
- **Heard**: Voice sounded clean and natural on evaluation sentences. However Phase 2 started diverging toward incorrect phoneme rows when evaluated on a diverse 5-sentence suite — resulting in an accent shift toward an unintended pattern.
- **Changed**: Decided to use Phase 1 (blend-only) output as the final submission since it is clean, natural, and definitively beats the baseline by +0.0186.

---

## Final Submitted voice.pt
- **Source**: Best convex blend from Run 2 / Run 4 Phase 1.
- **Weights**: af_sky 24.2%, bf_lily 48.7%, af_nova 24.7%, af_sarah 2.4%
- **Three-sentence similarity**: **0.6034** (American pipeline, improves on baseline 0.5848)
- **Heard**: Clear and natural speech, no artifacts, intelligibility high.
