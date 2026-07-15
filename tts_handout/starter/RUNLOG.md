# Run Log

### Run 1: Naive 50/50 Blend Baseline
- **Fitness Design**: Single-sentence cosine similarity on *"The quick brown fox..."* (default `blend.py`).
- **Settings**:
  - Voice selection: 50% `af_sarah` + 50% `bf_emma`.
- **Scores**:
  - Single-sentence similarity: **0.7963**
  - Three-sentence average similarity: **0.4878**
- **What was heard**: Standard, clear voice. It sounds generic and lacks the target speaker's unique accent and speech cadence.
- **What was changed**: Initial baseline run, nothing changed.

---

### Run 2: Convex Blend Weight Optimization (Top 5 Voices)
- **Fitness Design**: Average cosine similarity across 3 diverse sentences from reference transcripts to prevent sentence-length overfitting.
- **Settings**:
  - Search space: 5 weights (summing to 1) representing blend of `af_sarah`, `bf_emma`, `af_sky`, `bf_lily`, `af_nova`.
  - Iterations: 100.
- **Scores**:
  - Three-sentence average similarity: **0.6034**
- **What was heard**: A richer voice that blends features of multiple stock speakers. Timbre sounds significantly closer to the target speaker.
- **What was changed**: Expanded search space to top 5 stock voices and ran random walk on the weight simplex.

---

### Run 3: Broadcasted Perturbation with Soft L2 Penalty (Failed Run)
- **Fitness Design**: Cosine similarity minus a soft L2 regularization penalty (`reg_coef * L2_norm(perturbation)`).
- **Settings**:
  - Start point: Best blend from Run 2.
  - Search space: 256-dimensional perturbation vector added to all 510 rows.
  - Iterations: 150.
  - Step size: 0.02 (annealing down to 0.01).
  - Regularization coefficient: 0.05.
- **Scores**:
  - Three-sentence average similarity: **0.6034** (Stuck/no improvement).
- **What was heard**: The audio did not change from the baseline blend.
- **What was changed/Why it failed**: The soft L2 penalty added an immediate score disadvantage for any non-zero step, causing all random walk proposals to be rejected.

---

### Run 4: Broadcasted Perturbation with Strict Hard Clamping (Successful Run)
- **Fitness Design**: Pure cosine similarity across 3 evaluation sentences, with no soft penalty.
- **Settings**:
  - Start point: Best blend from Run 2.
  - Search space: 256-dimensional perturbation vector.
  - Clamping: Strictly clamp perturbation values to `[-0.05, 0.05]` to maintain speech naturalness.
  - Iterations: 150.
  - Step size: 0.02 (annealing down to 0.01).
- **Scores**:
  - Three-sentence average similarity: **[BEST_SCORE]**
  - Single-sentence similarity (fox text): **[SINGLE_SCORE]**
- **What was heard**: **[WHAT_WAS_HEARD]**
- **What was changed**: Removed the soft L2 penalty and replaced it with hard boundary constraints to allow free optimization within a safe range of naturalness.
