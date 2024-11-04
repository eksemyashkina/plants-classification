from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import json
import wandb
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import models
from models.resnet50 import ResNet
from models.mobilenet_v2 import MobileNetV2
from dataset import PlantsDataset
from utils import train_transform, test_transform, EMA


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on plant dataset")
    parser.add_argument("--train-root", type=str, default="data/plants/train", help="Path to the training data")
    parser.add_argument("--test-root", type=str, default="data/plants/test", help="Path to the testing data")
    parser.add_argument("--load-to-ram", type=bool, default=False, help="Load dataset to RAM")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and testing")
    parser.add_argument("--pin-memory", type=bool, default=True, help="Pin memory for DataLoader")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers for DataLoader")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weights-path", type=str, default="weights/mobilenet_v2-b0353104.pth", choices=["weights/resnet50-0676ba61.pth", "weights/mobilenet_v2-b0353104.pth"], help="Path to the pre-trained weights")
    parser.add_argument("--project-name", type=str, default="plants_classifier", help="WandB project name")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type")
    parser.add_argument("--criterion", type=str, default="CrossEntropyLoss", help="Loss function type")
    parser.add_argument("--labels-path", type=str, default="labels.json", help="Path to the labels json file")
    parser.add_argument("--max-norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the training on")
    parser.add_argument("--model", type=str, default="mobilenet", choices=["resnet", "mobilenet"], help="Model class name")
    parser.add_argument("--save-frequency", type=int, default=4, help="Frequency of saving model weights")
    parser.add_argument("--logs-dir", type=str, default="resnet-logs", choices=["resnet-logs", "mobilenet-logs"], help="???")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.labels_path, "r") as fp:
        labels = json.load(fp)
    num_classes = len(labels)

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(exist_ok=True)

    wandb.init(project=args.project_name, dir=logs_dir)

    train_dataset = PlantsDataset(root=args.train_root, load_to_ram=args.load_to_ram, transform=train_transform, labels=labels)
    test_dataset = PlantsDataset(root=args.test_root, load_to_ram=args.load_to_ram, transform=test_transform, labels=labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=args.pin_memory, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory, num_workers=args.num_workers)

    device = torch.device(args.device)

    if args.model == "resnet":
        model = ResNet(weights_path=args.weights_path)
        model.fc = nn.Linear(512 * model.expansion, num_classes)
        nn.init.xavier_uniform_(model.fc.weight)
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
        else:
            param.requires_grad = False
    elif args.model == "mobilenet":
        model = MobileNetV2(weights_path=args.weights_path)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        nn.init.xavier_uniform_(model.classifier[1].weight)
        for name, param in model.named_parameters():
            if "classifier" or "features.18" or "features.17" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    model = model.to(device)

    optimizer_class = getattr(torch.optim, args.optimizer)
    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate)
    criterion_class = getattr(nn, args.criterion)
    criterion = criterion_class()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    best_accuracy = 0

    train_loss_ema, train_accuracy_ema, grad_norm_ema = EMA(), EMA(), EMA()
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}/{args.num_epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm).item()
            optimizer.step()
            train_loss = loss.item()
            train_accuracy = (logits.argmax(dim=1) == labels).sum().item() / logits.shape[0]
            pbar.set_postfix({"loss": train_loss_ema(train_loss), "accuracy": train_accuracy_ema(train_accuracy), "grad_norm": grad_norm_ema(grad_norm)})
            wandb.log(
                {
                    "train/epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_accuracy,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm,
                }
            )

        model.eval()
        test_loss, test_accuracy = 0.0, 0.0
        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f"Val epoch {epoch}/{args.num_epochs}")
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                test_loss += loss.item()
                test_accuracy += (logits.argmax(dim=1) == labels).sum().item()
        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader.dataset)
        print(f"loss: {test_loss:.3f}, accuracy: {test_accuracy:.3f}")

        wandb.log(
            {
                "val/epoch": epoch,
                "val/test_loss": test_loss,
                "val/test_accuracy": test_accuracy,
            }
        )

        scheduler.step()

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), logs_dir / f"checkpoint-best-{epoch:09}.pth")
        elif epoch % args.save_frequency == 0:
            torch.save(model.state_dict(), logs_dir / f"checkpoint-{epoch:09}.pth")

    wandb.finish()


if __name__ == "__main__":
    main()
