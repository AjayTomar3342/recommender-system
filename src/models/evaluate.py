import torch


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for users, items, vectors, labels in dataloader:
            outputs = model(users, items, vectors).squeeze()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Model Accuracy: {accuracy:.2f}%")
    return accuracy