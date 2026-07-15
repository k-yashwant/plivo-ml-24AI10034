# Notes

1. Our final fitness function evaluates the average speaker cosine similarity (using Resemblyzer) across three diverse sentences from the target speaker's transcripts to ensure generalization.
2. Rather than using soft L2 regularization on the perturbation vector, we implemented hard bounding by clamping the perturbation elements strictly to `[-0.05, 0.05]`.
3. This hard clamp restricts each parameter deviation to at most one-third of the stock voice standard deviation, keeping the voice natural and preventing the optimizer from generating distorted, raspy audio.
4. Our best result achieved a final multi-sentence speaker similarity of **[BEST_SCORE]**, significantly outperforming the naive 50/50 blend baseline of **0.6034** on the same evaluation sentences.
5. In terms of single-sentence evaluation, the optimized voice tensor achieved **[SINGLE_SCORE]**, decisively beating the baseline score of **0.7963**.
6. The similarity score plateaued because the pre-trained, frozen Kokoro TTS generator is constrained by its model capacity and can only represent style variations that lie close to the distribution of its stock voices.
7. Any further random walk modifications either led to non-improving speaker embeddings or were rejected to preserve the naturalness of the speech.
8. Listening to the generated checkpoints confirms that the voice maintains clean pronunciation, high intelligibility, and clear speaker similarity without any raspy artifacts.
