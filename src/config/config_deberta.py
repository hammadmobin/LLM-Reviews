DEBERTA_MODEL_NAME = "microsoft/deberta-base"

DEBERTA_TOKENIZER_CONFIG = {
    "truncation": True,
    "padding": True,
    "max_length": 512
}

DEBERTA_TRAINING_ARGS = {
    "output_dir": "./results/deberta",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_dir": "./logs/deberta_logs",
    "logging_steps": 10
}
