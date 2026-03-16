import numpy as np
import time
from .vocabulary import Vocabulary
from .dataset import SkipGramDataset
from .model import Word2VecModel


class Trainer:
    """
    Manages the word2vec training loop.

    Learning rate schedule:
        Linear decay from lr_start to lr_min over the total number of steps.
        This matches the original word2vec C implementation and is important
        for convergence — a decaying LR prevents the model from oscillating
        around the minimum in later epochs.
    """

    def __init__(
            self,
            embed_dim: int = 100,
            window_size: int = 5,
            n_negatives: int = 5,
            lr_start: float = 0.025,
            lr_min: float = 0.0001,
            epochs: int = 5,
            min_count: int = 2,
            subsample_threshold: float = 1e-4,
            seed: int = 42,
    ):
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.n_negatives = n_negatives
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.epochs = epochs
        self.min_count = min_count
        self.subsample_threshold = subsample_threshold
        np.random.seed(seed)

        self.vocab: Vocabulary | None = None
        self.model: Word2VecModel | None = None

    def train(self, text: str) -> Word2VecModel:
        # ── Build vocabulary ─────────────────────────────────────────
        self.vocab = Vocabulary(self.min_count, self.subsample_threshold)
        token_ids = self.vocab.build(text)
        print(f"Vocab size: {self.vocab.vocab_size}")
        print(f"Corpus length (after min_count filter): {len(token_ids)}")

        # ── Subsampling ──────────────────────────────────────────────
        token_ids = self.vocab.subsample(token_ids)
        print(f"Corpus length (after subsampling): {len(token_ids)}")

        # ── Init model and dataset ───────────────────────────────────
        self.model = Word2VecModel(self.vocab.vocab_size, self.embed_dim)
        dataset = SkipGramDataset(self.vocab, self.window_size, self.n_negatives)

        # Pre-generate all pairs (fits in memory for typical corpora)
        all_pairs = list(dataset.get_pairs(token_ids))
        total_steps = len(all_pairs) * self.epochs
        print(f"Training pairs per epoch: {len(all_pairs):,}")
        print(f"Total steps: {total_steps:,}\n")

        step = 0
        for epoch in range(self.epochs):
            # Shuffle pairs each epoch for better convergence
            np.random.shuffle(all_pairs)

            epoch_loss = 0.0
            t_start = time.time()

            for center, context in all_pairs:
                # Linear LR decay
                lr = max(
                    self.lr_min,
                    self.lr_start * (1.0 - step / total_steps)
                )

                # Sample negatives (exclude center + context)
                negatives = dataset.sample_negatives(
                    exclude={center, context},
                    k=self.n_negatives
                )

                # Forward pass + gradient update
                loss = self.model.forward_and_grad(center, context, negatives, lr)
                epoch_loss += loss
                step += 1

            elapsed = time.time() - t_start
            avg_loss = epoch_loss / len(all_pairs)
            print(f"Epoch {epoch+1}/{self.epochs} | loss={avg_loss:.4f} | "
                  f"lr={lr:.5f} | time={elapsed:.1f}s")

        return self.model
