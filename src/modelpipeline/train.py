import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
from config import Config
from model import DeepFakeDetector
from dataset import build_dataloaders

def train():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on: {device}")

    # Load data and initialize model
    loaders, train_df = build_dataloaders(cfg)
    model = DeepFakeDetector(
        backbone_name=cfg.BACKBONE,
        pretrained=cfg.PRETRAINED,
        dropout=cfg.DROPOUT,
        num_classes=cfg.NUM_CLASSES,
    ).to(device)

    # Class weighting for imbalanced datasets
    n_real = (train_df["Label"] == "REAL").sum()
    n_fake = (train_df["Label"] == "FAKE").sum()
    total = n_real + n_fake
    class_weights = torch.tensor([total/n_real, total/n_fake], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.LABEL_SMOOTHING)

    # Differential Learning Rates
    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": cfg.LR / 10},
        {"params": model.classifier.parameters(), "lr": cfg.LR},
    ], weight_decay=cfg.WEIGHT_DECAY)

    # Scheduler setup
    warmup_epochs = 3
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    best_auc = 0.0
    for epoch in range(cfg.EPOCHS):
        # --- Standard FP32 Training Loop ---
        model.train()
        train_loss, correct, total_samples = 0.0, 0, 0
        
        train_bar = tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")

        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Forward pass (Standard FP32)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            # Backward pass (No Scaler)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Stats
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            train_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total_samples:.4f}")

        scheduler.step()

        # --- Validation ---
        auc = evaluate(model, loaders["val"], device)
        
        print(f"[*] Epoch {epoch+1} Results | Loss: {train_loss/len(loaders['train']):.4f} | Val AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"{cfg.CHECKPOINT_DIR}/best.pth")
            print(f"  [+] Saved new best model (AUC: {best_auc:.4f})")

def evaluate(model, loader, device):
    """Simplified evaluation without autocast."""
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  Validating", leave=False):
            imgs = imgs.to(device)
            logits = model(imgs)
            
            # probabilities for the positive class (FAKE)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    return roc_auc_score(all_labels, all_probs)

if __name__ == "__main__":
    train()
