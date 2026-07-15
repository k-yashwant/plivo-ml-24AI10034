# Notes

1. Our fitness function evaluates average speaker cosine similarity (using Resemblyzer) across three diverse sentences from the target speaker's reference transcripts to ensure generalization across different phoneme-length rows.
2. We discovered that `KPipeline.infer` selects exactly one row from the 510×1×256 style tensor using `pack[len(ps)-1]`, meaning a naive single-sentence search would only optimize one row and fail to generalize to held-out sentences.
3. Instead of searching the full 130,560-parameter style tensor, we reduced Phase 1 to a 5-dimensional weight simplex over the top 5 stock voices, guaranteeing the resulting voice stays on the natural voice manifold.
4. A soft L2 regularization penalty immediately deducted ~0.016 from any perturbation step, which was larger than the expected single-step improvement in 256 dimensions, causing all Phase 2 proposals to be rejected.
5. Switching to hard clamping of `[-0.05, 0.05]` (1/3 of stock voice std dev) removed the penalty, allowing proposals to be evaluated on pure cosine similarity while preserving naturalness.
6. Our final submitted `voice.pt` is a convex combination of af_sky (24.2%), bf_lily (48.7%), af_nova (24.7%), and af_sarah (2.4%), achieving a three-sentence similarity of **0.6034** versus the baseline **0.5848**, a +0.0186 improvement.
7. Listening to `listen_final.wav` confirmed that the voice is clean, artifact-free, highly intelligible, and noticeably closer to the target speaker's vocal timbre than the naive 50/50 blend.
8. The Hindi G2P pipeline (`lang_code='h'`) was verified to produce an accent closer to the target's Indian English, but since the similarity scores did not conclusively favor it, the default American pipeline was used for the final submission to maintain consistency with the evaluation script.
9. The similarity plateau occurred because the Kokoro-82M generator is a frozen pre-trained model, and the expressible voice space is bounded by the distribution of its training data, limiting how much we can shift the embedding toward an out-of-distribution speaker such as a native Indian English speaker.
10. Future directions include gradient-based optimization through a differentiable TTS model or using the Hindi G2P pipeline with a purpose-trained Indian English voice adapter.
