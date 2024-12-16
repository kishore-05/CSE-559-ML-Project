import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
import os

# Paths to your dataset (relative path since files are in the same folder as the code)
current_dir = os.path.dirname(os.path.realpath('tcrbert.py'))  # Get the directory of the current script
tcr_train_path = os.path.join(current_dir, 'train.csv')
tcr_test_path = os.path.join(current_dir, 'test.csv')

# Load the datasets directly from the CSV files
tcr_train = pd.read_csv(tcr_train_path, header=None) 
tcr_test = pd.read_csv(tcr_test_path, header=None)    

# Manually define column names
tcr_train.columns = ['tcr_sequence', 'binding', 'label']
tcr_test.columns = ['tcr_sequence', 'binding', 'label']

# Print column names and first few rows to verify
print("Training Data Columns:", tcr_train.columns)
print("Testing Data Columns:", tcr_test.columns)

# Inspect the first few rows to check the data
print(tcr_train.head())
print(tcr_test.head())

# Preprocessing: Select TCR sequences and labels, then drop any missing data
tcr_train_data = tcr_train[['tcr_sequence', 'label']].dropna()  
tcr_test_data = tcr_test[['tcr_sequence', 'label']].dropna()    

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(tcr_train_data)
test_dataset = Dataset.from_pandas(tcr_test_data)

# Load the pre-trained TCR-BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd')

# Tokenization function with reduced max length
def tokenize_function(examples):
    return tokenizer(examples['tcr_sequence'], truncation=True, padding='max_length', max_length=128)  # Reduced max length

# Apply tokenization to the datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Define the model (pre-trained TCR-BERT model)
model = BertForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd', num_labels=2)

# Prepare for training with reduced batch size and gradient accumulation
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Reduced batch size
    per_device_eval_batch_size=8,   # Reduced eval batch size
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=2,  
    fp16=True, 
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Print evaluation results
print(f"Evaluation Results: {eval_results}")

# Predictions and additional metrics
predictions = trainer.predict(test_dataset)
pred_labels = predictions.predictions.argmax(axis=-1)

accuracy = accuracy_score(tcr_test_data['label'], pred_labels)
precision, recall, f1, _ = precision_recall_fscore_support(tcr_test_data['label'], pred_labels, average='binary')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")