import re
from collections import Counter

class Vocabulary:
    """
    Builds a vocabulary from a raw text corpus.

    Subsampling of frequent words is implemented using Mikolov's formula:
        P(discard) = 1 - sqrt(t / freq)
    where t=1e-4 is the subsampling threshold. This mirrors the original C code
    and speeds up training by reducing the dominance of very common words.
    """

    def __init__(self, min_count: int = 2, subsample_threshold: float = 1e-4):
        self.min_count = min_count
        self.subsample_threshold = subsample_threshold

        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.word_freq: dict[str, int] = {}
        self.vocab_size: int = 0

    def build(self, text: str) -> list[int]:
        """Tokenise, filter by min_count, return token-id sequence."""
        tokens = re.findall(r'\b[a-z]+\b', text.lower())
        counts = Counter(tokens)

        # Filter rare words
        vocab_words = sorted(
            [w for w, c in counts.items() if c >= self.min_count]
        )
        self.word2idx = {w: i for i, w in enumerate(vocab_words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.word_freq = {w: counts[w] for w in vocab_words}
        self.vocab_size = len(vocab_words)

        # Convert corpus to ids (dropping unknown words)
        return [self.word2idx[t] for t in tokens if t in self.word2idx]

    def subsample(self, token_ids: list[int]) -> list[int]:
        """
        Probabilistically drop frequent tokens.
        Total corpus size is needed to compute per-word frequencies.
        """
        total = len(token_ids)
        keep = []
        for idx in token_ids:
            word = self.idx2word[idx]
            freq = self.word_freq[word] / total
            p_keep = (self.subsample_threshold / freq) ** 0.5
            if p_keep >= 1.0 or __import__('numpy').random.random() < p_keep:
                keep.append(idx)
        return keep
