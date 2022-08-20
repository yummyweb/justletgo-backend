import json
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

with open('conversations.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tags.append("depression")
    w = tokenize(intent[0])
    all_words.extend(w)
    xy.append((w, "depression"))

ignore_words = ['?', ',', '/', '.', '!']

all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))

X_train = []
y_train = []
for (p, tag) in xy:
    bag = bag_of_words(p, all_words)
    X_train.append(bag)

    label = tags.index(tag)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = words.to(labels)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}')

print(f'final loss = {loss.item():.4f}')