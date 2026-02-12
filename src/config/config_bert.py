BERT_MODEL_NAME = "bert-base-uncased"

BERT_TOKENIZER_CONFIG = {
    "truncation": True,
    "padding": True,
    "max_length": 512
}

BERT_TRAINING_ARGS = {
    "output_dir": "./results/bert",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_dir": "./logs/bert_logs",
    "logging_steps": 10
}
