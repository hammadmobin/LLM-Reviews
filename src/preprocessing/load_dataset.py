import pandas as pd

def load_authentic(path):
    return pd.read_csv(path)

def load_seed(path):
    return pd.read_csv(path)

def load_generated(path):
    return pd.read_csv(path, on_bad_lines="skip")

def load_synthetic14k(path):
    return pd.read_csv(path)

def load_synthetic_llm(path):
    return pd.read_csv(path)

def build_suspicious(seed_df, synthetic14k_df, generated_df):
    return pd.concat([seed_df, synthetic14k_df, generated_df], ignore_index=True)
