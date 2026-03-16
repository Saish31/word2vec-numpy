import numpy as np

class SkipGramDataset:
    """
    Generates (center, context) training pairs and negative samples.

    Negative Sampling Distribution:
        Words are sampled proportional to freq^(3/4), which is the
        unigram distribution raised to the 3/4 power used in the
        original word2vec paper. This compresses the distribution so
        that rare words are sampled more often than under the raw
        unigram distribution, but common words still dominate less.
    """

    def __init__(self, vocab, window_size: int = 5, n_negatives: int = 5):
        self.vocab = vocab
        self.window_size = window_size
        self.n_negatives = n_negatives

        # Build negative sampling table (freq^0.75)
        freq = np.array([
            vocab.word_freq[vocab.idx2word[i]]
            for i in range(vocab.vocab_size)
        ], dtype=np.float64)
        freq_powered = freq ** 0.75
        self.neg_probs = freq_powered / freq_powered.sum()

    def get_pairs(self, token_ids: list[int]):
        """Yield (center_idx, context_idx) pairs using dynamic window."""
        for i, center in enumerate(token_ids):
            # Dynamic window: sample window size uniformly in [1, window_size]
            # This matches the original paper and gives closer words more weight
            win = np.random.randint(1, self.window_size + 1)
            start = max(0, i - win)
            end = min(len(token_ids), i + win + 1)
            for j in range(start, end):
                if j != i:
                    yield center, token_ids[j]

    def sample_negatives(self, exclude: set[int], k: int) -> np.ndarray:
        """
        Sample k negative word indices, excluding the positive context word
        and the center word. Rejection sampling is used for correctness,
        which is fast in practice since vocab >> k.
        """
        negs = []
        while len(negs) < k:
            candidates = np.random.choice(
                self.vocab.vocab_size,
                size=k * 2,
                p=self.neg_probs
            )
            for c in candidates:
                if c not in exclude and len(negs) < k:
                    negs.append(c)
        return np.array(negs, dtype=np.int32)
