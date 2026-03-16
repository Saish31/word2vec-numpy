# Word2Vec — Pure NumPy Implementation

Skip-Gram with Negative Sampling (SGNS) implemented from scratch using only NumPy.

## Quickstart

```bash
pip install numpy
python train.py
```

## Architecture

| Component | Description |
|---|---|
| `Vocabulary` | Tokenisation, min-count filtering, frequency-based subsampling |
| `SkipGramDataset` | Skip-gram pair generation with dynamic window; neg sampling via freq^0.75 |
| `Word2VecModel` | Two weight matrices W_in, W_out; forward pass, loss, exact gradients, SGD |
| `Trainer` | Training loop with linear LR decay |

## Math: Loss & Gradients

**Objective (maximise):**

    J = log σ(v_c · v_o) + Σ_k log σ(−v_c · v_k)

**Gradients:**

    ∂loss/∂v_c = (σ(v_c·v_o) − 1)·v_o  +  Σ_k (1 − σ(−v_c·v_k))·v_k
    ∂loss/∂v_o = (σ(v_c·v_o) − 1)·v_c
    ∂loss/∂v_k = (1 − σ(−v_c·v_k))·v_c

## Key Design Decisions

- **Two separate matrices** (W_in, W_out): standard SGNS setup, empirically better than weight tying
- **freq^0.75 negative sampling**: compresses distribution so rare words are sampled more
- **Subsampling frequent words**: P(discard) = 1 − sqrt(t/freq), t=1e-4
- **Dynamic window**: window size sampled uniformly in [1, max_window] per training step
- **Linear LR decay**: lr decays from 0.025 → 0.0001 across total steps
- **No ML framework**: only `numpy`, `re`, `collections`, `time`, `os`
