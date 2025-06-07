#hypothetically would run this in the future with more representative data (oversampling + more data)
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch


def preprocess(df):
    df["text"] = (df["headline"].fillna("") + " " +
                  df["description"].fillna("") + " " +
                  df["instruction"].fillna("")).str.strip()
    df = df[df["event"].notna()]
    return df[["text", "event"]]

test_df = preprocess(pd.read_csv("/Users/ijeomaosakwe/Documents/GitHub/EmergenCease/data/extracted_data_test.csv"))
train_df = preprocess(pd.read_csv("/Users/ijeomaosakwe/Documents/GitHub/EmergenCease/data/extracted_data_train.csv"))
val_df = preprocess(pd.read_csv("/Users/ijeomaosakwe/Documents/GitHub/EmergenCease/data/extracted_data_val.csv"))


# --- 2. Label encoding ---
all_labels = pd.concat([train_df["event"], val_df["event"], test_df["event"]])
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

train_df["label"] = label_encoder.transform(train_df["event"])
val_df["label"] = label_encoder.transform(val_df["event"])
test_df["label"] = label_encoder.transform(test_df["event"])

# --- 3. Oversample minority classes in train set ---
max_size = train_df["label"].value_counts().max()
train_df = train_df.groupby("label").apply(
    lambda x: resample(x, replace=True, n_samples=max_size, random_state=42)
).reset_index(drop=True)

# --- 4. Convert to HuggingFace Datasets ---
train_ds = Dataset.from_pandas(train_df[["text", "label"]])
val_ds = Dataset.from_pandas(val_df[["text", "label"]])
test_ds = Dataset.from_pandas(test_df[["text", "label"]])

# --- 5. Tokenize ---
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_fn(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# --- 6. Load model ---
num_labels = len(label_encoder.classes_)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# --- 7. Training setup ---
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()


predictions = trainer.predict(test_ds)
pred_labels = predictions.predictions.argmax(-1)
true_labels = predictions.label_ids


print(classification_report(
    true_labels,
    pred_labels,
    labels=range(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
))


cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation=90, cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()