# You need to run this one, another time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from collections import Counter
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

# ------------------ Dataset -------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=100):
        self.texts = [self.tokenize(t) for t in texts]
        self.labels = torch.tensor(labels.values, dtype=torch.float32)
        self.max_len = max_len

        if vocab is None:
            counter = Counter(word for sent in self.texts for word in sent)
            self.vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common())}
            self.vocab["<PAD>"] = 0
            self.vocab["<UNK>"] = 1
        else:
            self.vocab = vocab

        self.texts = [self.encode(t) for t in self.texts]
        self.texts = pad_sequence(self.texts, batch_first=True, padding_value=0)[:, :max_len]

    def tokenize(self, text):
        return text.lower().split()

    def encode(self, tokens):
        return torch.tensor([self.vocab.get(t, 1) for t in tokens], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# ------------------ Models -------------------
class MLP(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 100, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        return self.fc(x)

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, kernel_size=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(nn.ReLU()(x)).squeeze(-1)
        return self.fc(x)

# ------------------ Train & Eval -------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    for X, y in tqdm(loader):
        X, y = X.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            output = model(X)
            preds = (output.cpu() > 0.5).int()
            y_true.extend(y.int().tolist())
            y_pred.extend(preds.squeeze(1).tolist())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, f1

# ------------------ Main -------------------
def main():
    df = pd.read_csv("../processed_data/MPDD.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        df['Prompt'], df['isMalicious'], test_size=0.2, random_state=42, stratify=df['isMalicious']
    )

    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test, vocab=train_dataset.vocab)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(train_dataset.vocab)

    models = {
        "MLP": MLP(vocab_size).to(device),
        "LSTM": LSTMModel(vocab_size).to(device),
        "CNN": CNNModel(vocab_size).to(device)
    }

    results = []

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        for epoch in range(5):
            print(f"Epoch {epoch+1}")
            train(model, train_loader, optimizer, criterion, device)

        acc, f1 = evaluate(model, test_loader, device)
        results.append({"Model": name, "Accuracy": acc, "F1": f1})

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results_deep_models.csv", index=False)

    print("\n=== Results ===")
    print(results_df)

    # Plot
    plt.figure(figsize=(10,5))
    sns.barplot(x='F1', y='Model', data=results_df.sort_values(by='F1', ascending=True))
    plt.title("Model Comparison (F1-score)")
    plt.savefig("deep_models_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
