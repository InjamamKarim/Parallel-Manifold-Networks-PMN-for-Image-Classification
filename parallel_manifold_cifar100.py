#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel Manifold Network (PMN) on CIFAR-100

- Small CNN backbone
- Two parallel 2D-manifold paths (radius / cos(phi))
- Each path: 2 manifold layers (+ LayerNorm + Dropout)
- Fused representation -> linear classifier
- Optional entropy-regularizer per "universe" (classifier column)

Usage (example):

    python parallel_manifold_cifar100.py \
        --data-dir ./data \
        --out-dir ./runs/pmn128 \
        --n-manifolds 128 \
        --epochs 40

"""

import os
import math
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def get_cifar100_loaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------
# Manifold Layer
# ---------------------------------------------------------------------

class ManifoldLayer(nn.Module):
    """
    Projects input h (B, D_in) to N 2D-manifolds (x,y).
    Then returns a 1D activation per manifold based on:

    - "radius": r = ||z||
    - "cos_phi": cos(phi) = cos(atan2(y,x))
    """

    def __init__(self, in_dim: int, n_manifolds: int, activation: str = "radius"):
        super().__init__()
        self.in_dim = in_dim
        self.n_manifolds = n_manifolds
        self.activation = activation
        # We always produce 2 * n_manifolds -> (x, y) for each
        self.fc = nn.Linear(in_dim, 2 * n_manifolds)

    def forward(self, h: torch.Tensor):
        """
        h: (B, in_dim)
        Returns:
            a: (B, n_manifolds)   - 1D activation per manifold
            (z, r, phi): internal geometry (for analysis/visualization)
        """
        B = h.size(0)
        z = self.fc(h)                # (B, 2N)
        z = z.view(B, self.n_manifolds, 2)  # (B, N, 2)

        x = z[..., 0]
        y = z[..., 1]

        r = torch.sqrt(x * x + y * y + 1e-8)
        phi = torch.atan2(y, x)

        if self.activation == "radius":
            a = r
        elif self.activation == "cos_phi":
            a = torch.cos(phi)
        else:
            a = r  # Fallback

        return a, (z, r, phi)


# ---------------------------------------------------------------------
# Parallel Manifold Network
# ---------------------------------------------------------------------

class ParallelManifoldNet(nn.Module):
    """
    Parallel Manifold Network on CIFAR-100

    Backbone:
        3 conv layers + adaptive avgpool -> (B, 256)

    Two paths:
        - radius path (N manifolds)
        - cos(phi) path (N manifolds)
        Each path: 2 manifold layers with LayerNorm + Dropout

    Fused:
        concat(radius, cos_path) -> (B, 2N)
        linear classifier -> (B, num_classes)
    """

    def __init__(
        self,
        n_manifolds: int = 128,
        num_classes: int = 100,
        dropout_p: float = 0.15,
    ):
        super().__init__()

        self.n_manifolds = n_manifolds

        # Small CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 256 x 1 x 1
        )

        feat_dim = 256
        N = n_manifolds

        # Radius path
        self.manifold1_r = ManifoldLayer(feat_dim, N, activation="radius")
        self.norm1_r = nn.LayerNorm(N)
        self.drop1_r = nn.Dropout(dropout_p)

        self.manifold2_r = ManifoldLayer(N, N, activation="radius")
        self.norm2_r = nn.LayerNorm(N)
        self.drop2_r = nn.Dropout(dropout_p)

        # Cos(phi) path
        self.manifold1_p = ManifoldLayer(feat_dim, N, activation="cos_phi")
        self.norm1_p = nn.LayerNorm(N)
        self.drop1_p = nn.Dropout(dropout_p)

        self.manifold2_p = ManifoldLayer(N, N, activation="cos_phi")
        self.norm2_p = nn.LayerNorm(N)
        self.drop2_p = nn.Dropout(dropout_p)

        # Classifier: 2N -> num_classes
        self.classifier = nn.Linear(2 * N, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)       # (B, 256, 1, 1)
        h = h.view(h.size(0), -1)  # (B, 256)

        # Radius path
        a1_r, _ = self.manifold1_r(h)
        a1_r = self.drop1_r(self.norm1_r(a1_r))
        a2_r, _ = self.manifold2_r(a1_r)
        a2_r = self.drop2_r(self.norm2_r(a2_r))

        # Cos(phi) path
        a1_p, _ = self.manifold1_p(h)
        a1_p = self.drop1_p(self.norm1_p(a1_p))
        a2_p, _ = self.manifold2_p(a1_p)
        a2_p = self.drop2_p(self.norm2_p(a2_p))

        # Fusion
        a = torch.cat([a2_r, a2_p], dim=1)  # (B, 2N)
        logits = self.classifier(a)
        return logits


# ---------------------------------------------------------------------
# Entropy regularization: encourages few strong classes per universe
# ---------------------------------------------------------------------

def manifold_entropy_regularizer(model: ParallelManifoldNet, eps: float = 1e-8) -> torch.Tensor:
    """
    Simple entropy regularizer on classifier weights.

    Interprets each column j of the classifier weight matrix as
    a distribution over classes via |W[c,j]|, and penalizes high entropy.

    Returns:
        scalar tensor (mean entropy over universes)
    """
    W = model.classifier.weight  # (C, U_eff = 2N)
    absW = W.abs() + eps
    col_sums = absW.sum(dim=0, keepdim=True)    # (1, U_eff)
    probs = absW / col_sums                     # (C, U_eff)
    entropy = -(probs * (probs + eps).log()).sum(dim=0)  # (U_eff,)
    return entropy.mean()


# ---------------------------------------------------------------------
# Training / Evaluation
# ---------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    correct1 = 0
    correct5 = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss_sum += loss.item() * y.size(0)
            total += y.size(0)

            # Top-1
            pred1 = logits.argmax(dim=1)
            correct1 += (pred1 == y).sum().item()

            # Top-5
            top5 = logits.topk(5, dim=1).indices
            correct5 += (top5.eq(y.view(-1, 1))).any(dim=1).sum().item()

    avg_loss = loss_sum / total
    top1 = correct1 / total
    top5 = correct5 / total
    return avg_loss, top1, top5


def train(
    args: argparse.Namespace,
):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, test_loader = get_cifar100_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = ParallelManifoldNet(
        n_manifolds=args.n_manifolds,
        num_classes=100,
        dropout_p=args.dropout,
    ).to(device)

    n_params = count_parameters(model)
    print(model)
    print(f"\nTrain: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)} | "
          f"Classes: 100 | Params: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_top1 = 0.0

    log_path = os.path.join(args.out_dir, "train_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Train Ep {epoch:03d}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)

            ce_loss = criterion(logits, y)

            if args.lambda_entropy > 0.0:
                ent = manifold_entropy_regularizer(model)
                loss = ce_loss + args.lambda_entropy * ent
            else:
                ent = torch.tensor(0.0, device=device)
                loss = ce_loss

            loss.backward()
            optimizer.step()

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            running_correct += (logits.argmax(dim=1) == y).sum().item()

        train_loss = running_loss / total
        train_acc = running_correct / total

        val_loss, val_top1, val_top5 = evaluate(model, test_loader, device, criterion)

        line = (f"Ep {epoch:03d} | "
                f"TrainLoss {train_loss:.4f} Acc {train_acc*100:.2f}% | "
                f"ValLoss {val_loss:.4f} Top1 {val_top1*100:.2f}% Top5 {val_top5*100:.2f}%")
        print(line)
        print(line, file=log_file)
        log_file.flush()

        if val_top1 > best_top1:
            best_top1 = val_top1
            ckpt_path = os.path.join(args.out_dir, "pmn_best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_top1": best_top1,
                    "args": vars(args),
                },
                ckpt_path,
            )

    log_file.close()
    print(f"\nDONE! Best Top-1: {best_top1*100:.2f}%")
    print(f"Log saved to: {log_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel Manifold Network on CIFAR-100"
    )
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Path to CIFAR-100 data directory")
    parser.add_argument("--out-dir", type=str, default="./runs/pmn",
                        help="Directory to save logs and checkpoints")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of dataloader workers")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--n-manifolds", type=int, default=128,
                        help="Number of manifolds per path (64, 128, 256, ...)")
    parser.add_argument("--dropout", type=float, default=0.15,
                        help="Dropout probability in manifold paths")
    parser.add_argument("--lambda-entropy", type=float, default=1e-3,
                        help="Weight for classifier entropy regularizer (per universe)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
