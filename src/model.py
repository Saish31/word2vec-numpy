import numpy as np

class Word2VecModel:
    """
    Skip-Gram with Negative Sampling (SGNS).

    Two weight matrices:
        W_in  : shape (V, D) — input (center word) embeddings
        W_out : shape (V, D) — output (context word) embeddings

    The final word vectors are taken from W_in after training.
    W_out is a secondary matrix used only during training.
    Using two separate matrices (rather than tied weights) is standard
    in word2vec and empirically produces better embeddings.

    ── Objective (per training pair) ───────────────────────────────────
    Maximise the log-likelihood:
        J = log σ(v_c · v_o)  +  Σ_{k=1}^{K} log σ(−v_c · v_k)

    where:
        v_c = W_in[center]       ← center word vector (shape: D)
        v_o = W_out[context]     ← positive context vector (shape: D)
        v_k = W_out[neg_k]       ← negative sample vectors (shape: D)
        σ   = sigmoid function

    We minimise loss = −J.

    ── Gradients ───────────────────────────────────────────────────────
    Let:
        s_pos = σ(v_c · v_o)
        s_neg = σ(−v_c · v_k)   for each negative k

    ∂loss/∂v_o  = (s_pos − 1) · v_c          [positive context gradient]
    ∂loss/∂v_k  = (1 − s_neg) · v_c          [negative context gradient]
    ∂loss/∂v_c  = (s_pos − 1) · v_o
                + Σ_k (1 − s_neg_k) · v_k    [center word gradient]

    These are exact closed-form gradients — no numerical approximation.
    Only the rows corresponding to center, context, and negative words
    are touched per step, giving O(D·K) cost instead of O(V·D) for
    full softmax.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        self.V = vocab_size
        self.D = embed_dim

        # Initialise with small random values (not zeros — symmetry breaking)
        # Uniform in [-0.5/D, 0.5/D] following the original C implementation
        scale = 0.5 / embed_dim
        self.W_in  = np.random.uniform(-scale, scale, (vocab_size, embed_dim))
        self.W_out = np.zeros((vocab_size, embed_dim))  # output init to 0

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid: clamp input to avoid overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def forward_and_grad(
            self,
            center: int,
            context: int,
            negatives: np.ndarray,
            lr: float
    ) -> float:
        """
        Performs one forward pass, computes loss and gradients, and
        applies SGD updates in-place.

        Returns:
            loss (float): scalar loss for this training example
        """
        v_c = self.W_in[center]               # (D,)
        v_o = self.W_out[context]             # (D,)
        V_n = self.W_out[negatives]           # (K, D)

        # ── Forward pass ─────────────────────────────────────────────
        s_pos = self.sigmoid(v_c @ v_o)       # scalar
        s_neg = self.sigmoid(-(V_n @ v_c))    # (K,)

        # ── Loss ─────────────────────────────────────────────────────
        # Numerically clamp to avoid log(0)
        eps = 1e-10
        loss = -np.log(s_pos + eps) - np.sum(np.log(s_neg + eps))

        # ── Gradients ────────────────────────────────────────────────
        # ∂loss/∂v_c: accumulate contributions from positive + all negatives
        grad_v_c = (s_pos - 1.0) * v_o + ((1.0 - s_neg)[:, None] * V_n).sum(axis=0)

        # ∂loss/∂v_o (positive context word)
        grad_v_o = (s_pos - 1.0) * v_c

        # ∂loss/∂v_k (negative context words), shape (K, D)
        grad_V_n = (1.0 - s_neg)[:, None] * v_c[None, :]

        # ── SGD updates (in-place) ───────────────────────────────────
        self.W_in[center]    -= lr * grad_v_c
        self.W_out[context]  -= lr * grad_v_o
        self.W_out[negatives] -= lr * grad_V_n

        return loss

    def get_embeddings(self) -> np.ndarray:
        """Return the final word embeddings (W_in)."""
        return self.W_in

    def most_similar(self, word: str, vocab, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Compute cosine similarity between a query word and all vocabulary
        words, return top_n most similar.
        """
        if word not in vocab.word2idx:
            return []
        idx = vocab.word2idx[word]
        query = self.W_in[idx]

        # Cosine similarity: normalise all vectors, then dot product
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True) + 1e-10
        normed = self.W_in / norms
        sims = normed @ (query / (np.linalg.norm(query) + 1e-10))

        # Exclude the query word itself
        sims[idx] = -1.0
        top_indices = np.argsort(sims)[::-1][:top_n]
        return [(vocab.idx2word[i], float(sims[i])) for i in top_indices]
