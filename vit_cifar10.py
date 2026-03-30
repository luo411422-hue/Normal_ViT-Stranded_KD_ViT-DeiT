
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from VIT import ViT
from data import CIFAR10
import matplotlib.pyplot as plt

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train() #将模型设为训练模式，启用 dropout 等特定层。
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()#清空上一轮的梯度。
        outputs = model(images)#前向传播，得到 logits
        loss = criterion(outputs, labels) #计算交叉熵损失。
        loss.backward() #反向传播，计算梯度。
        optimizer.step() #更新模型参数。

        running_loss += loss.item() * images.size(0) #累计损失，乘以 batch 大小以获得总损失。
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0) #累计总样本数。
        correct += predicted.eq(labels).sum().item() #累计正确预测的样本数。
        
        if (batch_idx + 1) % 20 == 0: #每处理 20 个 batch 输出一次当前的平均损失和准确率。
             avg_loss = running_loss / total
             avg_acc = 100.0 * correct / total
             print(f"  Batch [{batch_idx+1}/{len(dataloader)}] Processed {total} images, Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
    epoch_loss = running_loss / total #计算平均损失。
    epoch_acc = 100.0 * correct / total #计算准确率。
    return epoch_loss, epoch_acc


@torch.no_grad() #装饰器，表示在评估过程中不需要计算梯度，节省内存和计算资源。
def evaluate(model, dataloader, criterion, device):
    model.eval() #将模型设为评估模式，禁用 dropout 等特定层。
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        if (batch_idx + 1) % 20 == 0: #每处理 20 个 batch 输出一次当前的平均损失和准确率，方便监控评估过程。
            avg_loss = running_loss / total
            avg_acc = 100.0 * correct / total
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] Processed {total} images, Loss: {avg_loss:.4f}, Acc: {avg_acc:.2f}%")
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = CIFAR10.get_loader()

    model = ViT(
        in_channels=3,
        patch_size=4,
        emb_size=256,
        img_size=32,
        depth=6,
        n_classes=10,
        num_heads=8,
        drop_p=0.1,
        forward_drop_p=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    epochs = 20
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # 新增：用于记录指标的列表
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

         # 新增：保存到列表
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
            f"Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    torch.save(best_model_wts, "vit_cifar10_best.pth")
    model.load_state_dict(best_model_wts)
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)

    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Final Eval Loss: {final_loss:.4f}, Final Eval Accuracy: {final_acc:.2f}%")
    plt.figure(figsize=(12, 4)) #可视化
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accs, label='Train Acc')
    plt.plot(range(1, epochs+1), test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png')   # 保存图片
    plt.show()

if __name__ == "__main__":
    main()
