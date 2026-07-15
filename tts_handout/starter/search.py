"""Advanced Search: Convex blend optimization followed by broadcasted perturbation.
Runs a 2-phase optimization:
1. Optimize the blend weights of the top 5 stock voices.
2. Direct search with a broadcasted 256-dimensional perturbation vector.
"""
import argparse
import os
import random
import numpy as np
import torch
import soundfile as sf

import synth
import similarity as sim

# Evaluation sentences of varying lengths (including the fox text) to guarantee generalization
EVAL_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Honestly, the best part of the trip was the food. I still think about it sometimes.",
    "Hi there, thanks for waiting. Could you tell me a little more about the issue you're seeing?",
    "I grew up in a small town, but I've been living in the city for almost eight years now.",
    "We can reschedule your appointment to Tuesday afternoon if that works better for you."
]



def fitness(voice, target_emb, texts, reg_loss=0.0):
    """Evaluate speaker similarity on multiple sentences and apply regularization."""
    total = 0.0
    for t in texts:
        wav = synth.synthesize(t, voice)
        total += sim.similarity_to_target(wav, target_emb)
    # Average similarity minus any regularization penalty
    return (total / len(texts)) - reg_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference_dir", required=True)
    ap.add_argument("--iters_blend", type=int, default=100, help="Iterations for blend weight search")
    ap.add_argument("--iters_pert", type=int, default=150, help="Iterations for perturbation search")
    ap.add_argument("--step", type=float, default=0.02, help="Initial perturbation step size")
    ap.add_argument("--reg_coef", type=float, default=0.05, help="Regularization penalty on perturbation norm")
    ap.add_argument("--out", default="voice.pt")
    ap.add_argument("--listen_every", type=int, default=10)
    args = ap.parse_args()

    # 1. Load target speaker embedding
    target = sim.target_embedding(args.reference_dir)
    
    # 2. Get top 5 stock voices based on blend.py ranking
    stock = synth.stock_voices()
    top_names = ["af_sarah", "bf_emma", "af_sky", "bf_lily", "af_nova"]
    top_tensors = [stock[name] for name in top_names]
    
    print(f"Loaded top 5 stock voices: {top_names}")

    # ==========================================
    # PHASE 1: Optimize Blend Weights
    # ==========================================
    print("\n--- PHASE 1: Optimizing Convex Blend Weights ---")
    
    # Start with naive 50/50 blend of the top 2 as the initial best
    best_weights = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    best_voice = sum(w * v for w, v in zip(best_weights, top_tensors))
    best_f = fitness(best_voice, target, EVAL_SENTENCES)
    print(f"Initial blend (50/50 sarah/emma) fitness: {best_f:.4f}")

    for i in range(1, args.iters_blend + 1):
        # Propose new weights by adding Dirichlet-like noise or random perturbations on simplex
        proposal = best_weights + np.random.normal(0, 0.1, size=5)
        proposal = np.clip(proposal, 0.0, 1.0)
        if proposal.sum() > 0:
            proposal = proposal / proposal.sum()
        else:
            proposal = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
            
        cand_voice = sum(w * v for w, v in zip(proposal, top_tensors))
        f = fitness(cand_voice, target, EVAL_SENTENCES)
        
        if f > best_f:
            best_weights = proposal
            best_voice = cand_voice
            best_f = f
            print(f"Blend Iter {i:3d} | New best fitness: {best_f:.4f} | Weights: {best_weights}")

    print(f"Best blend weights found: {best_weights} with fitness {best_f:.4f}")
    
    # ==========================================
    # PHASE 2: Broadcasted Perturbation Search
    # ==========================================
    print("\n--- PHASE 2: Optimizing Broadcasted Perturbation ---")
    
    # We will search for a shared 256-dimensional vector to perturb the blend voice
    best_pert = torch.zeros(1, 1, 256)
    step_size = args.step
    accepted = 0
    
    for i in range(1, args.iters_pert + 1):
        # Step-size annealing: decay step size over iterations
        current_step = step_size * (1.0 - (i / args.iters_pert) * 0.5)
        
        # Propose a perturbation in the 256-dim space
        prop_pert = best_pert + current_step * torch.randn(1, 1, 256)
        
        # Keep perturbation within a strict box of [-0.05, 0.05] (1/3 of stock voice std dev)
        # This keeps the voice sounding natural without needing a soft penalty.
        prop_pert = torch.clamp(prop_pert, -0.05, 0.05)
        
        # Apply the broadcasted perturbation to all 510 rows
        cand_voice = best_voice + prop_pert
        
        # Evaluate fitness (using pure similarity score)
        f = fitness(cand_voice, target, EVAL_SENTENCES)
        
        if f > best_f:
            best_pert = prop_pert
            best_f = f
            best_voice = cand_voice
            accepted += 1
            print(f"Pert Iter {i:3d} | Accepted #{accepted} | Fitness: {best_f:.4f} | Pert L2 norm: {torch.norm(best_pert).item():.4f}")
            
            if accepted % args.listen_every == 0:
                sf.write(f"listen_{accepted}.wav", synth.synthesize(EVAL_SENTENCES[0], best_voice), synth.SR)
                print(f"  -> Wrote listen_{accepted}.wav (GO LISTEN)")
        elif i % 5 == 0:
            print(f"Pert Iter {i:3d} / {args.iters_pert} ... (Current best: {best_f:.4f}, step: {current_step:.4f})")

    # Save final optimized voice tensor
    torch.save(best_voice, args.out)
    
    # Generate final validation audio
    sf.write("listen_final.wav", synth.synthesize(EVAL_SENTENCES[0], best_voice), synth.SR)
    print(f"\nFinal optimized fitness: {best_f:.4f} -> Saved to {args.out}")
    print("Wrote listen_final.wav — PLEASE LISTEN TO IT TO VERIFY QUALITY!")



if __name__ == "__main__":
    main()

