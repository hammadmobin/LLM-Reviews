import numpy as np
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

from src.config.config_bert import (
    BERT_MODEL_NAME,
    BERT_TOKENIZER_CONFIG,
    BERT_TRAINING_ARGS
)

from src.transformers.model_loader import load_tokenizer, load_classifier
from src.datasets.dataset_builder import ReviewDataset
from src.transformers.evaluation import evaluate_predictions


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return evaluate_predictions(preds, labels)


def train_bert(X, y):
    tokenizer = load_tokenizer(BERT_MODEL_NAME)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.tolist(), y.tolist(), test_size=0.20, stratify=y
    )

    # tokenize dataset
    train_encodings = tokenizer(X_train, **BERT_TOKENIZER_CONFIG)
    test_encodings = tokenizer(X_test, **BERT_TOKENIZER_CONFIG)

    train_dataset = ReviewDataset(train_encodings, y_train)
    test_dataset = ReviewDataset(test_encodings, y_test)

    # model init
    model = load_classifier(BERT_MODEL_NAME)

    training_args = TrainingArguments(**BERT_TRAINING_ARGS)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer
