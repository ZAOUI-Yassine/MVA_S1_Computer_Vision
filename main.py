import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from model_factory import ModelFactory
from torch.cuda.amp import GradScaler

def opts() -> argparse.ArgumentParser:
    """Option handling function"""
    parser = argparse.ArgumentParser(description="Optimized Training Script for Fast Convergence")
    parser.add_argument("--data", type=str, default="sketch_recvis2024/sketch_recvis2024", help="Data folder")
    parser.add_argument("--model_name", type=str, default="basic_cnn", help="Model name")
    parser.add_argument("--batch-size", type=int, default=128, metavar="B")  
    parser.add_argument("--epochs", type=int, default=50, metavar="N")
    parser.add_argument("--lr", type=float, default=0.0001, metavar="LR") 
    parser.add_argument("--num_workers", type=int, default=4, metavar="NW")
    parser.add_argument("--experiment", type=str, default="experiment", metavar="E")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N")
    args = parser.parse_args()
    return args

def train(model, train_loader, optimizer, scaler, use_cuda, epoch, criterion, args):
    """Training loop"""
    model.train()
    correct = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

    train_accuracy = 100.0 * correct / len(train_loader.dataset)
    print(f"\nTrain set: Accuracy: {correct}/{len(train_loader.dataset)} ({train_accuracy:.2f}%)\n")
    return total_loss / len(train_loader), train_accuracy

def validation(model, val_loader, use_cuda, criterion):
    """Validation loop"""
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100.0 * correct / len(val_loader.dataset)
    print(
        f"\nValidation set: Average loss: {val_loss:.4f}, "
        f"Accuracy: {correct}/{len(val_loader.dataset)} ({val_accuracy:.2f}%)\n"
    )
    return val_loss, val_accuracy

def main():
    """Main Function for training"""
    args = opts()
    use_cuda = torch.cuda.is_available()
    print(f"Using {'GPU' if use_cuda else 'CPU'}")

    # Ensure experiment directory exists
    os.makedirs(args.experiment, exist_ok=True)

    # Get model and transformations from ModelFactory
    model, train_transform, val_transform = ModelFactory(args.model_name).get_all()

    if use_cuda:
        model.cuda()

    # Data loaders
    train_loader = DataLoader(
        datasets.ImageFolder(os.path.join(args.data, "train_images"), transform=train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets.ImageFolder(os.path.join(args.data, "val_images"), transform=val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer, loss function, and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, verbose=True)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Training and validation loop
    best_val_acc = 0.0
    patience = 10  # Early stopping patience
    patience_counter = 0
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, scaler, use_cuda, epoch, criterion, args)
        val_loss, val_acc = validation(model, val_loader, use_cuda, criterion)
        scheduler.step(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_file = os.path.join(args.experiment, "model_best.pth")
            torch.save(model.state_dict(), best_model_file)  # Save the best model
            print(f"Saved best model for epoch {epoch}.")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    main()
