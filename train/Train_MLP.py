"""Train a Morgan-fingerprint MLP with masked BCE and validation early stopping."""

import argparse
import random

import numpy as np
import torch
from torch import nn

from train.common import add_shared_arguments, create_run, evaluate, finish_run, load_splits, save_json


class MLP(nn.Module):
    """Map molecular fingerprints to independent label logits."""

    def __init__(self, bits, hidden, dropout, labels):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(bits, hidden), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, labels))

    def forward(self, x):
        return self.layers(x)


@torch.no_grad()
def predict(model, x, device):
    model.eval()
    return model(torch.from_numpy(x).to(device)).sigmoid().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=.4)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.0001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--positive-weight", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.hidden, args.batch_size, args.lr, args.bits) <= 0:
        parser.error("Epochs, patience, dimensions, batch size and learning rate must be positive.")
    if not 0 <= args.dropout < 1 or args.weight_decay < 0 or args.radius < 0:
        parser.error("Invalid dropout, weight decay or radius.")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    splits, labels, split_meta = load_splits(args.split, args.radius, args.bits)
    run = create_run("mlp", args, labels, split_meta)
    x, y, mask = splits["train"]
    observed_rows = mask.any(axis=1)
    dataset = torch.utils.data.TensorDataset(*(torch.from_numpy(a[observed_rows]) for a in (x, y, mask)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                        generator=torch.Generator().manual_seed(args.seed))
    model = MLP(args.bits, args.hidden, args.dropout, len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    positives = (y * mask).sum(axis=0)
    negatives = mask.sum(axis=0) - positives
    weights = np.clip(negatives / np.maximum(positives, 1), 1, 10)
    positive_weight = torch.tensor(weights, dtype=torch.float32, device=device) if args.positive_weight else None
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=positive_weight)
    best, stale, history, best_epoch = -1., 0, [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0., 0
        for batch, truth, observed in loader:
            batch, truth, observed = batch.to(device), truth.to(device), observed.to(device)
            optimizer.zero_grad()
            element_loss = loss_fn(model(batch), truth)
            loss = (element_loss * observed).sum() / observed.sum()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * int(observed.sum())
            count += int(observed.sum())
        vx, vy, vm = splits["val"]
        validation = evaluate(vy, predict(model, vx, device), vm, labels)
        ap = validation["macro_ap"]
        history.append({"epoch": epoch, "train_loss": total / count, "val_macro_ap": ap})
        print(f"epoch={epoch} train_loss={total/count:.4f} val_macro_ap={ap:.4f}", flush=True)
        if ap > best:
            best, stale, best_epoch = ap, 0, epoch
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "bits": args.bits, "hidden": args.hidden, "dropout": args.dropout,
                        "radius": args.radius, "labels": labels}, run / "model.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    saved = torch.load(run / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    names = ["train", "val"] + (["test"] if args.evaluate_test else [])
    for name in names:
        sx, sy, sm = splits[name]
        save_json(run / f"{name}.json", evaluate(sy, predict(model, sx, device), sm, labels))
    save_json(run / "history.json", history)
    save_json(run / "summary.json", {"best_epoch": best_epoch, "epochs_ran": len(history),
                                     "best_val_macro_ap": best, "device": device,
                                     "parameter_count": sum(p.numel() for p in model.parameters()),
                                     "test_evaluated": args.evaluate_test})
    finish_run(run)
    print(f"Saved run: {run}")


if __name__ == "__main__":
    main()
