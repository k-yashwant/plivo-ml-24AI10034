"""Verify which G2P accent (American 'a', British 'b', Hindi 'h') gets the best score."""
import os
import sys
import torch
import soundfile as sf

import synth
import similarity as sim

# Set Assets directory
os.environ['KOKORO_DIR'] = '../kokoro_assets'

EVAL_SENTENCES = [
    "Hi there, thanks for waiting. Could you tell me a little more about the issue you're seeing?",
    "I grew up in a small town, but I've been living in the city for almost eight years now.",
    "We can reschedule your appointment to Tuesday afternoon if that works better for you."
]

def evaluate_accent(lang_code, voice_path, target_emb):
    # Reset global pipeline to use the specified lang_code
    synth._PIPELINE = None
    pipe = synth.get_pipeline(lang_code)
    voice = synth.load_voice(voice_path)
    
    scores = []
    import numpy as np
    for s in EVAL_SENTENCES:
        chunks = [r.audio for r in pipe(s, voice=voice)]
        wav = torch.cat([c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in chunks]).detach().cpu().numpy().astype(np.float32)
        scores.append(sim.similarity_to_target(wav, target_emb))
        
    avg_score = sum(scores) / len(scores)
    return avg_score

def main():
    target = sim.target_embedding("../reference")
    
    print("--- EVALUATING ACCENTS ON blend_baseline.pt ---")
    import numpy as np
    for lang in ["a", "b", "h"]:
        try:
            score = evaluate_accent(lang, "blend_baseline.pt", target)
            print(f"Language Code '{lang}': average similarity = {score:.4f}")
            
            # Generate a sample audio to listen to using the correct pipeline
            synth._PIPELINE = None
            pipe = synth.get_pipeline(lang)
            voice = synth.load_voice("blend_baseline.pt")
            chunks = [r.audio for r in pipe("Honestly, the best part of the trip was the food. I still think about it sometimes.", voice=voice)]
            wav = torch.cat([c if isinstance(c, torch.Tensor) else torch.tensor(c) for c in chunks]).detach().cpu().numpy().astype(np.float32)
            sf.write(f"accent_{lang}.wav", wav, synth.SR)
            print(f"  -> Wrote accent_{lang}.wav (GO LISTEN)")
        except Exception as e:
            print(f"Language Code '{lang}' failed: {e}")


if __name__ == "__main__":
    main()
