import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score


def train_model(model, train_loader, val_loader, epochs, device, cfg, logger):

    # =========================
    # CHECKPOINT FOLDER (FIXED)
    # =========================
    save_path = "checkpoints"
    os.makedirs(save_path, exist_ok=True)

    # =========================
    # LOSS + OPTIMIZER
    # =========================
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["hyperparameters"]["learning_rate"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg["hyperparameters"]["scheduler_gamma"],
        patience=cfg["hyperparameters"]["scheduler_step_size"],
    )

    # =========================
    # BEST TRACKING
    # =========================
    best_loss = float("inf")
    best_acc = 0.0
    best_f1 = 0.0
    best_epoch = -1

    model_name = cfg["save_dir"]["experiment_name"]  # b1, b3, etc

    logger.info("Training Started")
    logger.info(f"Saving checkpoints to: {save_path}")

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(epochs):

        # =========================
        # TRAIN
        # =========================
        model.to(device)
        model.train()

        train_loss, correct, total = 0, 0, 0
        y_true_tr, y_pred_tr = [], []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")

        for x, y in loop:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = out.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            y_true_tr.extend(y.cpu().numpy())
            y_pred_tr.extend(preds.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = correct / total
        train_f1 = f1_score(y_true_tr, y_pred_tr, average="macro", zero_division=0)

        # =========================
        # VALIDATION
        # =========================
        model.eval()

        val_loss, correct, total = 0, 0, 0
        y_true_v, y_pred_v = [], []

        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]")

            for x, y in loop:
                x, y = x.to(device), y.to(device)

                out = model(x)
                loss = criterion(out, y)

                val_loss += loss.item()

                preds = out.argmax(dim=1)

                correct += (preds == y).sum().item()
                total += y.size(0)

                y_true_v.extend(y.cpu().numpy())
                y_pred_v.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = correct / total
        val_f1 = f1_score(y_true_v, y_pred_v, average="macro", zero_division=0)

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # =========================
        # LOGGING
        # =========================
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train loss {train_loss:.4f}, acc {train_acc:.4f}, f1 {train_f1:.4f} | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.4f}, f1 {val_f1:.4f} | "
            f"LR {lr:.2e}"
        )

        # =========================
        # SAVE BEST ONLY
        # =========================
        if val_loss < best_loss:

            best_loss = val_loss
            best_acc = val_acc
            best_f1 = val_f1
            best_epoch = epoch

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "best_acc": best_acc,
                "best_f1": best_f1
            }

            save_file = os.path.join(save_path, f"{model_name}.pth")

            torch.save(checkpoint, save_file)

            logger.info(f"NEW BEST SAVED -> {save_file}")

    # =========================
    # FINAL SUMMARY
    # =========================
    logger.info("====================================")
    logger.info("Training Finished")
    logger.info(
        f"Best Epoch: {best_epoch+1} | "
        f"Loss: {best_loss:.4f} | "
        f"Acc: {best_acc:.4f} | "
        f"F1: {best_f1:.4f}"
    )
    logger.info("====================================")