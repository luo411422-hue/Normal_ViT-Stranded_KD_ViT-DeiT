import copy
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from einops import repeat
from einops.layers.torch import Rearrange

from data import CassavaData
from GIT_vit import Transformer, pair
from ResNet import *


NUM_CLASSES = 5
IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 768
DEPTH = 12
HEADS = 12
MLP_DIM = 3072
DIM_HEAD = 64
DROPOUT = 0.1
EMB_DROPOUT = 0.1

BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 4.0
ALPHA = 0.5
HARD_DISTILLATION = False
SEED = 42
NUM_WORKERS = 2





def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DistilledViT(nn.Module):
    def __init__(
        self,
        *,
        image_size=224,
        patch_size=16,
        num_classes=5,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, (
            "Image dimensions must be divisible by the patch size."
        )
        assert pool == "cls", "DeiT style distillation is defined for cls pooling."

        num_patches = (image_height // patch_height) * (image_width // patch_width) # 196
        patch_dim = channels * patch_height * patch_width # 3*14*14

        self.to_patch_embedding = nn.Sequential(
           
                nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e'),
            )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.distill_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(  #encoder
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.cls_head = nn.Linear(dim, num_classes)
        self.distill_head = nn.Linear(dim, num_classes)

    def forward(self, img, return_distill=False):
        batch = img.shape[0]
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch)
        distill_tokens = repeat(self.distill_token, "1 1 d -> b 1 d", b=batch)
        #确保每个样本都有
        x = torch.cat((cls_tokens, distill_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : x.shape[1]]
        x = self.dropout(x)

        x = self.transformer(x) #encoder

        cls_logits = self.cls_head(x[:, 0]) #class token
        distill_logits = self.distill_head(x[:, 1]) #distillation token

        if return_distill:
            return cls_logits, distill_logits

        return (cls_logits + distill_logits) / 2


def create_teacher(num_classes):
    return ResNet56(dataset="cassava", num_classes=num_classes)


def load_teacher_weights(model, weight_path, device): #得到老师训练的权重
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Teacher weights not found: {weight_path}\n"
            "Please train the teacher first so that resnet56_cassava_best.pth exists."
        )

    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model


def deit_distillation_loss(
    cls_logits,
    distill_logits,
    teacher_logits,
    labels,
    alpha,
    temperature,
    hard_distillation=False,
):
    cls_loss = F.cross_entropy(cls_logits, labels)

    if hard_distillation: #硬蒸馏 只学习老师的最终决策 logits 里面的最大值就是老师的最终决策（one-hot)
        teacher_labels = teacher_logits.argmax(dim=1)
        distill_loss = F.cross_entropy(distill_logits, teacher_labels)
    else: #软蒸馏 学习老师的softmax概率
        distill_loss = F.kl_div(
            F.log_softmax(distill_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature ** 2)

    total_loss = (1.0 - alpha) * cls_loss + alpha * distill_loss
    return total_loss, cls_loss, distill_loss


def train_one_epoch(student, teacher, dataloader, optimizer, device):
    student.train()
    teacher.eval()

    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_distill_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            teacher_logits = teacher(images)

        optimizer.zero_grad()
        cls_logits, distill_logits = student(images, return_distill=True)
        total_loss, cls_loss, distill_loss = deit_distillation_loss(
            cls_logits,
            distill_logits,
            teacher_logits,
            labels,
            alpha=ALPHA,
            temperature=TEMPERATURE,
            hard_distillation=HARD_DISTILLATION,
        )
        total_loss.backward()#计算梯度
        optimizer.step() #更新权重

        fused_logits = (cls_logits + distill_logits) / 2
        batch_size = images.size(0)
        running_total_loss += total_loss.item() * batch_size
        running_cls_loss += cls_loss.item() * batch_size
        running_distill_loss += distill_loss.item() * batch_size
        _, predicted = torch.max(fused_logits, dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 20 == 0:
            avg_total = running_total_loss / total
            avg_cls = running_cls_loss / total
            avg_distill = running_distill_loss / total
            avg_acc = 100.0 * correct / total
            print(
                f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Processed {total} images, Total Loss: {avg_total:.4f}, "
                f"Cls Loss: {avg_cls:.4f}, Distill Loss: {avg_distill:.4f}, "
                f"Acc: {avg_acc:.2f}%"
            )

    epoch_total_loss = running_total_loss / total
    epoch_cls_loss = running_cls_loss / total
    epoch_distill_loss = running_distill_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_total_loss, epoch_cls_loss, epoch_distill_loss, epoch_acc


@torch.no_grad()
def evaluate(student, dataloader, device):
    student.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = student(images)
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
        img_size=IMAGE_SIZE,
        val_ratio=0.2,
        seed=SEED,
        num_workers=NUM_WORKERS,
    )
    label_map = CassavaData.get_label_map()
    print("Cassava classes:")
    for label_id, label_name in sorted(label_map.items(), key=lambda item: int(item[0])):
        print(f"  {label_id}: {label_name}")
    
    TEACHER_WEIGHTS_PATH = Path(__file__).resolve().with_name("resnet50_cassava_best.pth")
    teacher = create_teacher(NUM_CLASSES)
    teacher = load_teacher_weights(teacher, TEACHER_WEIGHTS_PATH, device).to(device)
    for parameter in teacher.parameters(): #冻结权重，教师网络是提前训练好的，权重无需改变
        parameter.requires_grad = False

    student = DistilledViT(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        dim=EMBED_DIM,
        depth=DEPTH,
        heads=HEADS,
        mlp_dim=MLP_DIM,
        dim_head=DIM_HEAD,
        dropout=DROPOUT,
        emb_dropout=EMB_DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_student_wts = copy.deepcopy(student.state_dict())
    
    STUDENT_SAVE_PATH = Path(__file__).resolve().with_name("deit_b16_cassava_best.pth")
    print(f"Teacher weights: {TEACHER_WEIGHTS_PATH}")
    print(f"Student save path: {STUDENT_SAVE_PATH}")
    print(
        f"DeiT config -> alpha: {ALPHA}, temperature: {TEMPERATURE}, "
        f"hard_distillation: {HARD_DISTILLATION}"
    )

    for epoch in range(EPOCHS):
        train_total_loss, train_cls_loss, train_distill_loss, train_acc = train_one_epoch(
            student, teacher, train_loader, optimizer, device
        )
        val_loss, val_acc = evaluate(student, val_loader, device)
        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Total Loss: {train_total_loss:.4f} "
            f"Train Cls Loss: {train_cls_loss:.4f} "
            f"Train Distill Loss: {train_distill_loss:.4f} "
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
    print(f"Best DeiT student saved to: {STUDENT_SAVE_PATH.resolve()}")


if __name__ == "__main__":
    main()
