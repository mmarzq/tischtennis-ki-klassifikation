import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset, random_split

class Tischtennis1DCNN(nn.Module):
    def __init__(self, input_channels=10, num_classes=4, window_size=200):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def load_data():
    X = np.load('processed_data/X_minimal.npy')
    y = np.load('processed_data/y_minimal.npy')
    with open('processed_data/info_minimal.json', 'r') as f:
        info = json.load(f)
    class_names = info['stroke_types']
    # PyTorch expects (batch, channels, seq_len)
    X = np.transpose(X, (0, 2, 1)).astype(np.float32)
    y = y.astype(np.int64)
    return X, y, class_names

def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        if (epoch+1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_cnn_pytorch.pth')
    print("Training abgeschlossen. Bestes Val Acc: {:.4f}".format(best_val_acc))

def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    return running_loss / total, correct / total

def main():
    X, y, class_names = load_data()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    n_total = len(dataset)
    n_test = int(0.2 * n_total)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_test - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Tischtennis1DCNN(input_channels=X.shape[1], num_classes=len(class_names), window_size=X.shape[2]).to(device)
    #print("Windowsgröße:", X.shape[2])
    print("Starte Training für 50 Epochen...")
    train_model(model, train_loader, val_loader, device, epochs=300, lr=0.001)
    print("Evaluierung auf Testdaten...")
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
