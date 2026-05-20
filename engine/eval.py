import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate(model, loader, device, class_names, logger):

    model.eval()
    model.to(device)

    preds_all, labels_all = [], []
    losses = []

    criterion = torch.nn.CrossEntropyLoss()

    # =========================
    # EVALUATION LOOP
    # =========================
    with torch.no_grad():

        pbar = tqdm(loader, desc="🔍 Evaluating", leave=True)

        for x, y in pbar:

            x, y = x.to(device), y.to(device)

            out = model(x)

            loss = criterion(out, y)

            losses.append(loss.item())

            preds = out.argmax(1)

            preds_all += preds.cpu().tolist()
            labels_all += y.cpu().tolist()

            # =========================
            # UPDATE PROGRESS BAR
            # =========================
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

    # =========================
    # METRICS
    # =========================
    loss = np.mean(losses)
    acc = accuracy_score(labels_all, preds_all)

    # =========================
    # LOG RESULTS
    # =========================
    logger.info("====================================")
    logger.info(" TEST RESULTS")
    logger.info(f"Loss: {loss:.4f} | Acc: {acc:.4f}")
    logger.info("====================================")

    print(f"\n✅ Loss: {loss:.4f} | Acc: {acc:.4f}")

    # =========================
    # CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(labels_all, preds_all)

    ConfusionMatrixDisplay(
        cm,
        display_labels=class_names
    ).plot(
        cmap="Blues",
        xticks_rotation=45
    )

    plt.title("Confusion Matrix")
    plt.show()

    return loss, acc

