import os
import math
import time
import json
import random
import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from transformers import BlipImageProcessor, get_cosine_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from dataset import VQAGenDataset
from model import VQAGenModel

# -----------------------------
# Repro & cuDNN
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True

# -----------------------------
# Configs (CLI-friendly)
# -----------------------------
@dataclass
class TrainConfig:
    csv_path: str = "/kaggle/input/vivqa/ViVQA-main/ViVQA-main/train.csv"
    image_folder: str = "/kaggle/input/vivqa/drive-download-20220309T020508Z-001/train"
    checkpoint_dir: str = "/kaggle/input/checkpoint/transformers/default/1/checkpoints"
    save_dir: str = "/kaggle/working/checkpoints"

    batch_size: int = 4
    accum_steps: int = 8                # 4 * 8 = effective batch 32
    num_epochs: int = 60
    val_split: float = 0.1
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    base_lr: float = 2e-4               # for fusion/decoder/text
    vision_lr: float = 1e-5             # smaller to protect ViT
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    warmup_ratio: float = 0.06          # % of total steps for warmup
    use_amp: bool = True
    resume_epoch: int = 0

    # Progressive unfreezing
    stage1_epochs: int = 8              # fusion+decoder
    stage2_epochs: int = 12             # + text encoder
    # remaining epochs -> stage3: + vision(last block)

    # Early stopping
    es_patience: int = 6
    es_min_delta: float = 1e-4

    # Logging/plots
    log_csv: str = "train_log.csv"
    curve_png: str = "training_curve.png"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str)
    p.add_argument("--image_folder", type=str)
    p.add_argument("--num_epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--accum_steps", type=int)
    p.add_argument("--base_lr", type=float)
    p.add_argument("--vision_lr", type=float)
    p.add_argument("--resume_epoch", type=int)
    args = p.parse_args()
    return args

# -----------------------------
# Stage utils
# -----------------------------
def set_stage(model: VQAGenModel, stage: int):
    """
    stage=1: train fusion + decoder only
    stage=2: + unfreeze text encoder
    stage=3: + unfreeze vision encoder (last block)
    """
    for p in model.parameters():
        p.requires_grad = False

    # Always train fusion + decoder
    for p in model.decoder.parameters():
        p.requires_grad = True
    if hasattr(model, "fusion"):
        for p in model.fusion.parameters():
            p.requires_grad = True
    # If using cross-attn version, also unfreeze cross_attn + projections
    if hasattr(model, "cross_attn"):
        for p in model.cross_attn.parameters():
            p.requires_grad = True
    if hasattr(model, "vision_proj"):
        for p in model.vision_proj.parameters():
            p.requires_grad = True
    if hasattr(model, "text_proj"):
        for p in model.text_proj.parameters():
            p.requires_grad = True

    if stage >= 2:
        for p in model.text_encoder.parameters():
            p.requires_grad = True

    if stage >= 3:
        # unfreeze last block of ViT (BLIP ViT encoder)
        try:
            last_block = model.vision_encoder.encoder.layers[-1]
            for p in last_block.parameters():
                p.requires_grad = True
        except Exception:
            # fallback: unfreeze entire vision if structure differs
            for p in model.vision_encoder.parameters():
                p.requires_grad = True

def build_optimizer(model: VQAGenModel, cfg: TrainConfig):
    # different LR for vision vs others
    vision_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "vision_encoder" in n:
            vision_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": cfg.base_lr})
    if vision_params:
        param_groups.append({"params": vision_params, "lr": cfg.vision_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    return optimizer

def count_trainable_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------------
# Train / Val loops
# -----------------------------
def run_one_epoch(model, loader, optimizer, scaler, device, cfg, scheduler=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    steps = 0

    pbar = tqdm(loader, disable=False, leave=False)
    optimizer_zero = optimizer is not None
    accum_steps = cfg.accum_steps if train else 1

    for step, batch in enumerate(pbar):
        pixel_values, input_ids, attention_mask, labels = batch
        pixel_values = pixel_values.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if cfg.use_amp and train:
                with autocast(dtype=torch.float16):
                    loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
                    loss = loss / accum_steps
                scaler.scale(loss).backward()
            else:
                loss, _ = model(pixel_values, input_ids, attention_mask, labels=labels)
                loss = loss / accum_steps
                loss.backward()

        if train:
            if (step + 1) % accum_steps == 0:
                if cfg.use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

        running_loss += loss.item() * accum_steps
        steps += 1
        pbar.set_description(f"{'Train' if train else 'Val'} loss: {running_loss/steps:.4f}")

    return running_loss / max(steps, 1)

# -----------------------------
# Plot curve
# -----------------------------
def plot_curves(csv_path, out_png):
    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)

# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(42)
    cfg = TrainConfig()
    # Optional CLI overrides
    try:
        args = parse_args()
        if args.csv_path: cfg.csv_path = args.csv_path
        if args.image_folder: cfg.image_folder = args.image_folder
        if args.num_epochs: cfg.num_epochs = args.num_epochs
        if args.batch_size: cfg.batch_size = args.batch_size
        if args.accum_steps: cfg.accum_steps = args.accum_steps
        if args.base_lr: cfg.base_lr = args.base_lr
        if args.vision_lr: cfg.vision_lr = args.vision_lr
        if args.resume_epoch is not None: cfg.resume_epoch = args.resume_epoch
    except SystemExit:
        # when running in notebooks w/o args
        pass

    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Dataset & Dataloaders
    vision_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")
    full_dataset = VQAGenDataset(cfg.csv_path, cfg.image_folder, vision_processor)

    val_size = int(len(full_dataset) * cfg.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )

    # Model
    model = VQAGenModel().to(device)

    # Resume?
    if cfg.resume_epoch > 0:
        model_path = os.path.join(cfg.checkpoint_dir, "best_model.pth")
        print(f"[INFO] Resuming weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))

    # Logs
    log_path = os.path.join(cfg.save_dir, cfg.log_csv)
    if not os.path.exists(log_path):
        pd.DataFrame(columns=["epoch","stage","trainable_params","lr","train_loss","val_loss","best_val","es_counter"]).to_csv(log_path, index=False)

    # Progressive stages
    total_epochs = cfg.num_epochs
    stage1_end = cfg.stage1_epochs
    stage2_end = cfg.stage1_epochs + cfg.stage2_epochs
    # stage3: remainder

    best_val = float("inf")
    es_counter = 0

    scaler = GradScaler(enabled=cfg.use_amp)

    global_step = 0
    for epoch in range(cfg.resume_epoch, total_epochs):
        # decide stage
        if epoch < stage1_end:
            stage = 1
        elif epoch < stage2_end:
            stage = 2
        else:
            stage = 3

        set_stage(model, stage)
        optimizer = build_optimizer(model, cfg)

        # total steps for scheduler in THIS epoch
        train_steps_per_epoch = math.ceil(len(train_loader) / cfg.accum_steps)
        # We use per-epoch cosine w/ warmup across full training:
        # define total training steps across all epochs for smoother schedule
        total_train_steps = train_steps_per_epoch * (total_epochs - epoch)
        warmup_steps = max(1, int(total_train_steps * cfg.warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps
        )

        trainable_params = count_trainable_params(model)
        current_lr = optimizer.param_groups[0]["lr"]

        # Train & Val
        train_loss = run_one_epoch(model, train_loader, optimizer, scaler, device, cfg, scheduler, train=True)
        val_loss   = run_one_epoch(model, val_loader, optimizer=None, scaler=scaler, device=device, cfg=cfg, scheduler=None, train=False)

        # Early stopping
        improved = (best_val - val_loss) > cfg.es_min_delta
        if improved:
            best_val = val_loss
            es_counter = 0
            # Save best
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_model.pth"))
            # Optimizer/scheduler states are not critical across stages (different heads),
            # keep code simple and robust.
            print(f"[INFO] New best @ epoch {epoch+1}: val={val_loss:.4f}")
        else:
            es_counter += 1

        # Save last (light)
        torch.save(model.state_dict(), os.path.join(cfg.save_dir, "last_model.pth"))

        # Append logs
        row = {
            "epoch": epoch + 1,
            "stage": stage,
            "trainable_params": trainable_params,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val": best_val,
            "es_counter": es_counter
        }
        df = pd.read_csv(log_path)
        df.loc[len(df)] = row
        df.to_csv(log_path, index=False)

        print(f"[EPOCH {epoch+1}/{total_epochs}] "
              f"Stage={stage} | params={trainable_params/1e6:.2f}M | "
              f"LR={current_lr:.2e} | Train={train_loss:.4f} | Val={val_loss:.4f} | "
              f"Best={best_val:.4f} | ES={es_counter}/{cfg.es_patience}")

        if es_counter >= cfg.es_patience:
            print("[INFO] Early stopping triggered.")
            break

    # Save tokenizers once if present in dataset
    try:
        full_dataset.q_tokenizer.save_pretrained(os.path.join(cfg.save_dir, "phobert_tokenizer"))
        full_dataset.a_tokenizer.save_pretrained(os.path.join(cfg.save_dir, "vit5_tokenizer"))
    except Exception:
        pass

    # Plot curves
    try:
        plot_curves(log_path, os.path.join(cfg.save_dir, cfg.curve_png))
        print(f"[INFO] Curves saved to {os.path.join(cfg.save_dir, cfg.curve_png)}")
    except Exception as e:
        print(f"[WARN] Plot failed: {e}")

if __name__ == "__main__":
    main()
