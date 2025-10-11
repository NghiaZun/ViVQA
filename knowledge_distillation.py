import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------------
# Giả sử bạn đã có:
# - teacher_model (pretrained, frozen)
# - student_model (kiến trúc tiếng Việt bạn đang có)
# - train_loader (DataLoader cho tập train VQA)
# ------------------------

# Loss KD (KL divergence giữa teacher & student)
def distillation_loss(student_logits, teacher_logits, T=2.0, alpha=0.5):
    """
    student_logits: [batch, num_classes]
    teacher_logits: [batch, num_classes]
    T: temperature
    alpha: trọng số giữa CE loss và KD loss
    """
    # Soft targets từ teacher
    soft_teacher = F.log_softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)

    # KL-divergence
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)

    return kd_loss

# Loss CE thông thường
ce_loss = nn.CrossEntropyLoss()

# ------------------------
# Huấn luyện
# ------------------------
def train_kd(student_model, teacher_model, train_loader, optimizer, device,
             alpha=0.5, T=2.0, num_epochs=5):
    student_model.train()
    teacher_model.eval()  # Teacher cố định, không update weight

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            images, questions, answers = batch
            images, questions, answers = images.to(device), questions.to(device), answers.to(device)

            # Forward
            with torch.no_grad():
                teacher_logits = teacher_model(images, questions)  # teacher output

            student_logits = student_model(images, questions)      # student output

            # Loss = CE + KD
            loss_ce = ce_loss(student_logits, answers)
            loss_kd = distillation_loss(student_logits, teacher_logits, T=T, alpha=alpha)

            loss = alpha * loss_ce + (1 - alpha) * loss_kd

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {total_loss/len(train_loader):.4f}")

    return student_model
