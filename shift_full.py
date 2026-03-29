import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODELS

class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class LargeNet(nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# BIAS FUNCTION

def add_bias(img, label, strength):
    img = img.clone()
    if label < 5:
        img[0] += strength  # red channel bias
    return torch.clamp(img, 0, 1)

# DATA

transform = transforms.Compose([
    transforms.ToTensor()
])

train_data_base = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=transform)

test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# TRAIN FUNCTION

def train_model(model, loader, epochs=3):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        correct, total = 0, 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        print(f"Epoch {ep+1}, Train Acc: {100*correct/total:.2f}")

    return model

# EVALUATE

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    return 100 * correct / total

# EXPERIMENT LOOP

bias_strengths = [0.0, 0.2, 0.5, 1.0]

small_train_accs = []
small_test_accs = []

large_train_accs = []
large_test_accs = []

for strength in bias_strengths:

    print("\n==========================")
    print("Bias Strength:", strength)
    print("==========================")

    # fresh dataset copy
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=False, transform=transform)

    # apply bias
    for i in range(len(train_data)):
        img, label = train_data[i]
        biased_img = add_bias(img, label, strength)
        train_data.data[i] = (biased_img.permute(1,2,0).numpy() * 255).astype(np.uint8)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    # -------- SMALL MODEL --------
    print("\nSmall Model:")
    small_model = train_model(SmallNet(), train_loader)

    small_train_acc = evaluate(small_model, train_loader)
    small_test_acc = evaluate(small_model, test_loader)

    print("Train Acc:", small_train_acc)
    print("Test Acc:", small_test_acc)

    small_train_accs.append(small_train_acc)
    small_test_accs.append(small_test_acc)

    # -------- LARGE MODEL --------
    print("\nLarge Model:")
    large_model = train_model(LargeNet(), train_loader)

    large_train_acc = evaluate(large_model, train_loader)
    large_test_acc = evaluate(large_model, test_loader)

    print("Train Acc:", large_train_acc)
    print("Test Acc:", large_test_acc)

    large_train_accs.append(large_train_acc)
    large_test_accs.append(large_test_acc)

# PLOTTING

plt.figure(figsize=(8,6))

plt.plot(bias_strengths, small_test_accs, marker='o', label="Small Model (Test)")
plt.plot(bias_strengths, large_test_accs, marker='o', label="Large Model (Test)")

plt.xlabel("Bias Strength")
plt.ylabel("Test Accuracy")
plt.title("Distribution Shift vs Model Performance")

plt.legend()
plt.savefig("final_shift_results.png")

print("\nSaved: final_shift_results.png")
