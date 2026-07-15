"""Verify the final optimized voice.pt against blend_baseline.pt."""
import os
import sys
import torch

import synth
import similarity as sim

# Set Assets directory
os.environ['KOKORO_DIR'] = '../kokoro_assets'

EVAL_SENTENCES = [
    "Hi there, thanks for waiting. Could you tell me a little more about the issue you're seeing?",
    "I grew up in a small town, but I've been living in the city for almost eight years now.",
    "We can reschedule your appointment to Tuesday afternoon if that works better for you."
]

FOX_TEXT = "The quick brown fox jumps over the lazy dog."


def evaluate_all(lang_code, baseline, optimized, target):
    # Reset global pipeline
    synth._PIPELINE = None
    pipe = synth.get_pipeline(lang_code)
    
    import numpy as np
    
    # 3-sentence evaluation
    base_scores = []
    opt_scores = []
    for s in EVAL_SENTENCES:
        # Baseline
        chunks_base = [r.audio for r in pipe(s, voice=baseline)]
        wav_base = torch.cat([c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in chunks_base]).detach().cpu().numpy().astype(np.float32)
        base_scores.append(sim.similarity_to_target(wav_base, target))
        
        # Optimized
        chunks_opt = [r.audio for r in pipe(s, voice=optimized)]
        wav_opt = torch.cat([c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in chunks_opt]).detach().cpu().numpy().astype(np.float32)
        opt_scores.append(sim.similarity_to_target(wav_opt, target))
        
    avg_base = sum(base_scores) / len(base_scores)
    avg_opt = sum(opt_scores) / len(opt_scores)
    
    # Fox text evaluation
    chunks_base_fox = [r.audio for r in pipe(FOX_TEXT, voice=baseline)]
    wav_base_fox = torch.cat([c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in chunks_base_fox]).detach().cpu().numpy().astype(np.float32)
    score_base_fox = sim.similarity_to_target(wav_base_fox, target)
    
    chunks_opt_fox = [r.audio for r in pipe(FOX_TEXT, voice=optimized)]
    wav_opt_fox = torch.cat([c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in chunks_opt_fox]).detach().cpu().numpy().astype(np.float32)
    score_opt_fox = sim.similarity_to_target(wav_opt_fox, target)
    
    print(f"\n=== EVALUATION RESULTS FOR LANG_CODE: '{lang_code}' ===")
    print(f"3-Sentence Baseline Average:  {avg_base:.4f}")
    print(f"3-Sentence Optimized Average: {avg_opt:.4f}")
    print(f"3-Sentence Improvement:       {avg_opt - avg_base:+.4f}")
    
    print(f"\nFox Text Baseline Score:      {score_base_fox:.4f}")
    print(f"Fox Text Optimized Score:     {score_opt_fox:.4f}")
    print(f"Fox Text Improvement:          {score_opt_fox - score_base_fox:+.4f}")


def main():
    target = sim.target_embedding("../reference")
    
    # Load tensors
    baseline = synth.load_voice("blend_baseline.pt")
    optimized = synth.load_voice("voice.pt")
    
    # Run evaluation for 'a' and 'h'
    evaluate_all("a", baseline, optimized, target)
    evaluate_all("h", baseline, optimized, target)


if __name__ == "__main__":
    main()

