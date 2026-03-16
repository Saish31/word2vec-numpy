"""
Entry point: downloads a small text corpus (text8 first 10MB slice),
trains word2vec, and runs a similarity evaluation.
"""

import urllib.request
import os
from src.vocabulary import Vocabulary
from src.trainer import Trainer


CORPUS_URL = "https://mattmahoney.net/dc/text8.zip"
CORPUS_PATH = "text8.zip"
TEXT_PATH = "text8"


def download_corpus():
    if not os.path.exists(TEXT_PATH):
        print("Downloading text8 corpus...")
        urllib.request.urlretrieve(CORPUS_URL, CORPUS_PATH)
        import zipfile
        with zipfile.ZipFile(CORPUS_PATH, 'r') as z:
            z.extractall(".")
        print("Downloaded.")


def main():
    download_corpus()

    with open(TEXT_PATH, "r") as f:
        # Use first 5M characters for fast demo (~1M tokens)
        text = f.read(5_000_000)

    trainer = Trainer(
        embed_dim=100,
        window_size=5,
        n_negatives=5,
        lr_start=0.025,
        lr_min=0.0001,
        epochs=5,
        min_count=5,
        subsample_threshold=1e-4,
    )

    model = trainer.train(text)

    # ── Evaluation: nearest neighbours ───────────────────────────────
    print("\n── Similarity Evaluation ──")
    test_words = ["king", "paris", "computer", "language", "music"]
    for word in test_words:
        results = model.most_similar(word, trainer.vocab, top_n=5)
        if results:
            neighbours = ", ".join(f"{w}({s:.3f})" for w, s in results)
            print(f"  {word:12s} → {neighbours}")
        else:
            print(f"  {word}: not in vocabulary")


if __name__ == "__main__":
    main()
