import pandas as pd

from src.datasets.load_dataset import (
    load_authentic,
    load_seed,
    load_generated,
    load_synthetic14k,
    build_suspicious
)

from src.baselines.baseline_runner import run_all_baselines
from src.transformers.train_deberta import train_deberta
from src.transformers.train_bert import train_bert


def main():

    print("\n============== Loading Data ==============")

    authentic = load_authentic("data/authentic_dataframe.csv")
    seed = load_seed("data/seed_reviews.csv")
    generated = load_generated("data/generated_reviews.csv")
    synth14k = load_synthetic14k("data/synthetic_14k_final.csv")

    suspicious = build_suspicious(seed, synth14k, generated)

    print("Authentic:", authentic.shape)
    print("Suspicious:", suspicious.shape)

    # RUN BASELINES
    print("\n============== Running Baselines ==============")
    run_all_baselines(authentic, suspicious)

    # PREP INPUT FOR TRANSFORMERS
    X = pd.concat([authentic["text"], suspicious["text"]], ignore_index=True).astype(str)
    y = pd.Series([0] * len(authentic) + [1] * len(suspicious))

    # RUN DEBERTA
    print("\n============== Training DeBERTa ==============")
    train_deberta(X, y)

    # RUN BERT
    print("\n============== Training BERT ==============")
    train_bert(X, y)


if __name__ == "__main__":
    main()
