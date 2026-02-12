from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def load_classifier(model_name, num_labels=2):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
