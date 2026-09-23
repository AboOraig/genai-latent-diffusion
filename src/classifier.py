"""
A small CNN classifier trained once on the real dataset, then reused purely
as an EVALUATION tool: given a batch of generated (decoded) images and the
class labels they were conditioned on, `classification_fidelity` reports
what fraction the classifier agrees are that class -- a standard proxy for
conditional generation quality when no pretrained Inception network /
FID pipeline is available (appropriate here since latents are tiny custom
VAE embeddings, not natural-image features an ImageNet-pretrained network
would recognize).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 28->14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 14->7
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc(h)


def train_classifier(train_loader, test_loader, device, epochs=3, lr=1e-3):
    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
        acc = evaluate_accuracy(model, test_loader, device)
        print(f"  [classifier] epoch {epoch+1}/{epochs}  test_acc={acc:.4f}")
    return model


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def classification_fidelity(model, images, target_labels, device):
    """Fraction of `images` the classifier assigns to `target_labels`
    (the labels they were generated/conditioned on)."""
    model.eval()
    images, target_labels = images.to(device), target_labels.to(device)
    pred = model(images).argmax(-1)
    return (pred == target_labels).float().mean().item()
