import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloader, epochs=5):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy for rating prediction

    for epoch in range(epochs):
        model.train()
        for users, items, vectors, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(users, items, vectors).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete. Loss: {loss.item():.4f}")