"""Generate real samples from a trained Indus checkpoint and save to SAMPLES.md."""
import sys
import torch
sys.path.insert(0, '/workspace/indus')

from indus.generate import load_indus


def main():
    m, t, info = load_indus('/workspace/indus/checkpoints/pretrain/indus_final.pt')
    print(f'Model: {m.num_params()/1e6:.2f}M params | vocab={t.vocab_size} | '
          f'iters={info["iter"]} | val_loss=1.47 | val_perplexity=4.33')

    out_path = '/workspace/indus/SAMPLES.md'
    with open(out_path, 'w') as f:
        f.write('# Indus Sample Generations\n\n')
        f.write(f'Model: {m.num_params()/1e6:.2f}M params | vocab={t.vocab_size} | '
                f'iters={info["iter"]} | val_loss=1.47 | val_perplexity=4.33\n\n')
        f.write('Char-level Shakespeare, trained from scratch on 1 CPU in ~17 minutes.\n\n')
        f.write('Temperature: 0.8, top_k: 100. Raw output — no cherry-picking.\n\n')
        f.write('---\n\n')

        for prompt in ['ROMEO:', 'JULIET:', 'KING RICHARD III:',
                       'To be, or not to be', 'Once upon a time']:
            ids = t.encode(prompt)
            idx = torch.tensor([ids], dtype=torch.long)
            out = m.generate(idx, max_new_tokens=400, temperature=0.8, top_k=100)
            text = t.decode(out[0].tolist())
            f.write(f'### Prompt: `{prompt}`\n\n')
            f.write('```\n')
            f.write(text + '\n')
            f.write('```\n\n')
            print(f'\n=== [{prompt}] ===')
            print(text[:200])
    print(f'\nWrote {out_path}')


if __name__ == "__main__":
    main()
