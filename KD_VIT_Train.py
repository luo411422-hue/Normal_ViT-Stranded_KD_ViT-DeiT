import copy
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import CassavaData
from ResNet import *
from VIT import ViT
try:
    from torchvision.models import ViT_B_16_Weights, vit_b_16
except ImportError:
    ViT_B_16_Weights = None
    from torchvision.models import vit_b_16


NUM_CLASSES = 5
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 4.0
ALPHA = 0.7
SEED = 42
NUM_WORKERS = 2
USE_PRETRAINED_STUDENT = True





def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_student(num_classes, use_pretrained=True):
    if use_pretrained:
        if ViT_B_16_Weights is not None:
            student = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            student = vit_b_16(pretrained=True)
    else:
        try:
            student = vit_b_16(weights=None)
        except TypeError:
            student = vit_b_16(pretrained=False)

    in_features = student.heads.head.in_features
    student.heads.head = nn.Linear(in_features, num_classes)
    return student



def load_teacher_weights(model, weight_path, device):
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Teacher weights not found: {weight_path}\n"
            "Please train the teacher first with resnet_cassava.py or place the teacher .pth file here."
        )

    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model


def kd_loss(student_logits, teacher_logits, labels, alpha, temperature):
    ce = F.cross_entropy(student_logits, labels)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    log_probs = F.log_softmax(student_logits / temperature, dim=1)
    KL_loss = F.kl_div(log_probs, soft_targets, reduction="batchmean") * (temperature ** 2)
    total = alpha * KL_loss + (1.0 - alpha) * ce
    return total, ce, KL_loss


def train_one_epoch(student, teacher, dataloader, optimizer, device, alpha, temperature):
    student.train()
    teacher.eval()

    running_total_loss = 0.0
    running_ce_loss = 0.0
    running_kd_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            teacher_logits = teacher(images)

        optimizer.zero_grad()
        student_logits = student(images)
        total_loss, ce_loss, distill_loss = kd_loss(
            student_logits, teacher_logits, labels, alpha, temperature
        )
        total_loss.backward() #计算梯度
        optimizer.step() #更新权重

        batch_size = images.size(0) #
        running_total_loss += total_loss.item() * batch_size # 累计总损失
        running_ce_loss += ce_loss.item() * batch_size # 累计交叉熵损失
        running_kd_loss += distill_loss.item() * batch_size # 累计蒸馏损失
        _, predicted = torch.max(student_logits, dim=1) # 获取预测结果
        total += labels.size(0) # 累计总样本数
        correct += predicted.eq(labels).sum().item() # 累计正确预测数

        if (batch_idx + 1) % 20 == 0:
            avg_total = running_total_loss / total
            avg_ce = running_ce_loss / total
            avg_kd = running_kd_loss / total
            avg_acc = 100.0 * correct / total
            print(
                f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Processed {total} images, Total Loss: {avg_total:.4f}, "
                f"CE: {avg_ce:.4f}, KD: {avg_kd:.4f}, Acc: {avg_acc:.2f}%"
            )

    epoch_total_loss = running_total_loss / total
    epoch_ce_loss = running_ce_loss / total
    epoch_kd_loss = running_kd_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_total_loss, epoch_ce_loss, epoch_kd_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(logits, dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = CassavaData.get_loaders(
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        val_ratio=0.2,
        seed=SEED,
        num_workers=NUM_WORKERS,
    )
    label_map = CassavaData.get_label_map()
    print("Cassava classes:")
    for label_id, label_name in sorted(label_map.items(), key=lambda item: int(item[0])):
        print(f"  {label_id}: {label_name}")

    TEACHER_WEIGHTS_PATH = Path(__file__).resolve().with_name("resnet50_cassava_best.pth")
    teacher = ResNet56(dataset="cassava", num_classes=5).to(device)
    teacher = load_teacher_weights(teacher, TEACHER_WEIGHTS_PATH, device).to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    # student = create_student(NUM_CLASSES, use_pretrained=USE_PRETRAINED_STUDENT).to(device)
    student = ViT(
        in_channels=3,
        patch_size=32,
        emb_size=256,
        img_size=224,
        depth=6,
        n_classes=5,
        num_heads=8, 
        drop_p=0.1,
        forward_drop_p=0.1,
    ).to(device)
  
    optimizer = optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_student_wts = copy.deepcopy(student.state_dict())

    STUDENT_SAVE_PATH = Path(__file__).resolve().with_name("kd_vit_b16_cassava_best.pth")
    print(f"Teacher weights: {TEACHER_WEIGHTS_PATH}")
    print(f"Student save path: {STUDENT_SAVE_PATH}")
    print(
        f"KD config -> alpha: {ALPHA}, temperature: {TEMPERATURE}, "
        f"pretrained_student: {USE_PRETRAINED_STUDENT}"
    )

    for epoch in range(EPOCHS):
        train_total_loss, train_ce_loss, train_kd_loss, train_acc = train_one_epoch(
            student,
            teacher,
            train_loader,
            optimizer,
            device,
            alpha=ALPHA,
            temperature=TEMPERATURE,
        )
        val_loss, val_acc = evaluate(student, val_loader, device)
        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Total Loss: {train_total_loss:.4f} "
            f"Train CE: {train_ce_loss:.4f} "
            f"Train KD: {train_kd_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_student_wts = copy.deepcopy(student.state_dict())
            torch.save(best_student_wts, STUDENT_SAVE_PATH)

    student.load_state_dict(best_student_wts)
    final_loss, final_acc = evaluate(student, val_loader, device)

    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Final Eval Loss: {final_loss:.4f}, Final Eval Accuracy: {final_acc:.2f}%")
    print(f"Best student model saved to: {STUDENT_SAVE_PATH.resolve()}")


if __name__ == "__main__":
    main()
