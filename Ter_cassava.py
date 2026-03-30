import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from data import CassavaData
from ResNet import *


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 20 == 0:
            avg_loss = running_loss / total
            avg_acc = 100.0 * correct / total
            print(
                f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Processed {total} images, Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%"
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = CassavaData.get_loaders(
        batch_size=64,
        img_size=224,
        val_ratio=0.2,
        seed=42,
        num_workers=2,
    )

    label_map = CassavaData.get_label_map()
    print("Cassava classes:")
    for label_id, label_name in sorted(label_map.items(), key=lambda item: int(item[0])):
        print(f"  {label_id}: {label_name}")

    model = ResNet56(dataset='cassava', num_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    epochs = 10
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    save_path = Path(__file__).resolve().with_name("resnet50_cassava_best.pth")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)

    model.load_state_dict(best_model_wts)
    final_loss, final_acc = evaluate(model, val_loader, criterion, device)

    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Final Eval Loss: {final_loss:.4f}, Final Eval Accuracy: {final_acc:.2f}%")
    print(f"Best model saved to: {save_path.resolve()}")


if __name__ == "__main__":
    main()
