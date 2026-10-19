# MyAI: a tiny AI that learns from what you type

`MyAI.swift` is one file of pure Swift + Foundation (no libraries, no GPU). It builds a small
decoder-only Transformer (the GPT design) from scratch: tokenizer, attention, hand-written
backpropagation, Adam, training, generation, and **online learning**. When you teach it a sentence, it
takes a few gradient steps on that sentence right away.

## How to run

```sh
swiftc -O MyAI.swift -o myai                # build (~7 s)
./myai                                      # self checks + train on built-in text + demo (~65 s)
./myai generate "my cat is named"           # continue a prompt (uses myai_model.json)
./myai teach "my bird is named kiwi."       # teach one sentence, shows before/after probability
./myai chat                                 # type lines to teach it; "? words" just asks; "quit"
./myai train mytext.txt                     # train a fresh model on your own text
# options: --steps 1500  --seed 1  --temp 0.7
```

## The 13 build steps

| Step | What it does | CME 295 concept |
|------|--------------|-----------------|
| 1 | seeded random numbers (same seed = same run) | reproducibility |
| 2 | byte-level BPE tokenizer: text <-> token IDs | tokenization, BPE |
| 3 | building blocks: Linear, LayerNorm, GELU, softmax | linear layer, LayerNorm, FFN |
| 4 | the model: token + position embeddings, 2 Transformer blocks, head | embeddings, decoder-only |
| 5 | forward pass, causal self-attention, cross-entropy loss | self-attention, causal mask |
| 6 | backward pass (hand-written gradients) | backpropagation |
| 7 | Adam optimizer with gradient clipping | Adam, learning rate |
| 8 | training loop on random text windows | next-token prediction |
| 9 | generation with temperature and guarded top-k | decoding, temperature, top-k |
| 10 | save / load weights as JSON | checkpoints |
| 11 | learning from new inputs (online fine-tuning with replay) | fine-tuning, forgetting |
| 12 | self checks that print PASS / FAIL | gradient check, evaluation |
| 13 | built-in training text + commands | training data |

## What a test run shows

- 100,565 learnable numbers; 1,500 training steps in about 62 s on one CPU core; loss 6.45 -> 0.11.
- Checks: tokenizer round trip, softmax sums to 1, determinism, gradient check (worst relative error 1.9e-7),
  causal mask, loss decreases, learns from input, reproduces the taught fact, save/load round trip. All PASS.
- Teaching "my dog is named biscuit.": P(" biscuit." | "my dog is named") goes from 5.8e-31 to 0.91, and
  `generate("my dog is named")` writes "my dog is named biscuit."

## Honest limits

It is tiny and trained on about 3 KB of text, so it memorizes and recombines patterns. It does not
understand or reason. Teaching one fact can disturb similar ones: after teaching "biscuit", "my cat is named"
came out as "my cat is named b zoe."
