import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import csv
import os

#######################
# Optional Memory Monitoring
#######################
try:
    import psutil
    def monitor_memory():
        process = psutil.Process(os.getpid())
        print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")
except ImportError:
    def monitor_memory():
        pass
    print("psutil not found. Memory monitoring disabled.")

#######################
# Iterable Dataset
#######################
class TCRDataset(IterableDataset):
    """
    A PyTorch IterableDataset that streams data from a large CSV file.
    Assumes each row: tcr_sequence,binding,label
    where label is '0' or '1'.
    """
    def __init__(self, file_path, max_len=100, hash_size=256):
        super(TCRDataset, self).__init__()
        self.file_path = file_path
        self.max_len = max_len
        self.hash_size = hash_size

    def process_line(self, line):
        # line: [tcr_sequence, binding, label]
        tcr_sequence = line[0]
        label_str = line[2]
        
        # Convert label directly to integer
        label_idx = int(label_str)

        # Hash and pad TCR sequence
        seq_features = [hash(ord(c)) % self.hash_size for c in tcr_sequence[:self.max_len]]
        if len(seq_features) < self.max_len:
            seq_features += [0] * (self.max_len - len(seq_features))

        X = torch.tensor(seq_features, dtype=torch.long)
        y = torch.tensor(label_idx, dtype=torch.long)
        return X, y

    def __iter__(self):
        with open(self.file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                yield self.process_line(row)


#######################
# Model Definition
#######################
class SimpleTCRModel(nn.Module):
    def __init__(self, hash_size=256, embed_dim=16, max_len=100, num_classes=2):
        super(SimpleTCRModel, self).__init__()
        self.embedding = nn.Embedding(hash_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * max_len, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, max_len)
        embeds = self.embedding(x)          # (batch_size, max_len, embed_dim)
        embeds = embeds.view(embeds.size(0), -1)
        out = self.relu(self.fc1(embeds))
        out = self.fc2(out)
        return out

#######################
# Training and Evaluation Functions
#######################
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(y_batch.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    return accuracy, precision, recall, f1

#######################
# Main Execution
#######################

if __name__ == "__main__":
    # Example usage:
    # Update these file paths to your large CSV files
    train_file = 'train.csv'
    test_file = 'test.csv'

    # Create datasets
    train_dataset = TCRDataset(train_file, max_len=100, hash_size=256)
    test_dataset = TCRDataset(test_file, max_len=100, hash_size=256)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=1024, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1024, num_workers=4)

    # Initialize model and training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTCRModel(hash_size=256, embed_dim=16, max_len=100, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    monitor_memory()

    # Train for a few epochs
    epochs = 100
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        monitor_memory()

        acc, prec, rec, f1 = evaluate(model, test_loader, device)
        print(f"Test Metrics - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    monitor_memory()