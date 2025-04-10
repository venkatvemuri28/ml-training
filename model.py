import os
import pandas as pd
from datasets import Dataset, load_metric
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# ----- CONFIGURATION -----

# Set the model name to use as a starting point.
# T5 is a versatile text-to-text model, which is well suited for summarization tasks.
model_name = "t5-small"

# Path to your CSV file with ticket data.
# The CSV should have at least two columns: one with the ticket's text and one with the summary.
data_path = "ticket_data.csv"

# Define the column names in your CSV (adjust if needed)
ticket_column = "ticket_text"
summary_column = "ticket_summary"

# Training parameters - feel free to adjust these.
training_args = TrainingArguments(
    output_dir="./t5_ticket_summary_model",
    evaluation_strategy="steps",
    eval_steps=500,             # How often to evaluate the model during training
    logging_steps=100,
    save_steps=500,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    learning_rate=5e-5,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=False,                # Set to True if you have a compatible GPU
)

# ----- DATA LOADING & PREPROCESSING -----

# Load your CSV data using pandas
df = pd.read_csv(data_path)

# Optional: If your CSV is large or you only want a fraction for testing,
# you can slice the dataframe, e.g., df = df.sample(n=1000, random_state=42)

# Convert the pandas DataFrame to a Hugging Face Datasets Dataset
dataset = Dataset.from_pandas(df)

# Split the dataset into training and evaluation sets
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Load the tokenizer from the chosen model.
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Set a maximum length for inputs and outputs.
max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    # T5 treats all tasks as text-to-text, so we prepend a task prefix (optional)
    inputs = ["summarize: " + str(text) for text in examples[ticket_column]]
    targets = [str(summary) for summary in examples[summary_column]]
    
    # Tokenize the inputs and targets.
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    
    # Tokenize targets with the tokenizer as well.
    # The label tokens should be shifted inside the model, so we just encode them.
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length")
    
    # Replace all padding token ids in labels by -100 (to ignore in loss computation)
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_seq]
        for labels_seq in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# ----- MODEL SETUP -----

# Load the model
model = T5ForConditionalGeneration.from_pretrained(model_name)

# The DataCollator will handle batching and dynamic padding for you
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Optionally, define a metric for evaluation. Here we use ROUGE for summarization tasks.
rouge_metric = load_metric("rouge")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    # Decode predictions and labels, skipping special tokens.
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we used it to ignore padding.
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute ROUGE scores
    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract the median scores for better stability
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return result

# ----- TRAINING -----

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Save the final model (+ tokenizer and config) to a directory.
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

print("Training complete. The model has been saved to:", training_args.output_dir)
