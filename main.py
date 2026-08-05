"""Train and evaluate CMAF-DDI with stratified cross-validation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import CMAFDDI
from utils import CMAFData, classification_metrics, configure_logging, seed_everything


LOGGER = logging.getLogger("cmafddi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce CMAF-DDI cross-validation experiments."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/DRKG"))
    parser.add_argument("--ddi-file", type=Path, default=None)
    parser.add_argument(
        "--entity-embedding-file",
        "--entity_embedding_file",
        dest="entity_embedding_file",
        type=Path,
        default=None,
        help="Knowledge-graph embedding matrix (.npy).",
    )
    parser.add_argument(
        "--graph-embedding-file",
        "--graph_embedding_file",
        dest="graph_embedding_file",
        type=Path,
        default=None,
        help="Molecular-graph embedding matrix (.npy).",
    )
    parser.add_argument(
        "--protein-embedding-file",
        "--protein_embedding_file",
        dest="protein_embedding_file",
        type=Path,
        default=None,
        help="Protein-sequence embedding matrix (.npy).",
    )

    parser.add_argument("--task", choices=("multiclass", "binary"), default="multiclass")
    parser.add_argument("--fusion", choices=("cmaf", "concat", "sum"), default="cmaf")
    parser.add_argument("--entity-dim", type=int, default=100)
    parser.add_argument("--structure-dim", type=int, default=300)
    parser.add_argument("--protein-dim", type=int, default=320)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--ffp-dim", type=int, default=2460)
    parser.add_argument("--classifier-hidden-1", type=int, default=2048)
    parser.add_argument("--classifier-hidden-2", type=int, default=2048)

    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--split-seed", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device name such as cpu, cuda, or cuda:0 (default: auto).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic PyTorch algorithms where available.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/cmaf-ddi")
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Limit batches per epoch for smoke tests; 0 uses every batch.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Limit evaluation batches for smoke tests; 0 uses every batch.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=0,
        help="Limit folds for smoke tests; 0 uses every fold.",
    )
    parser.add_argument("--no-save-checkpoints", action="store_true")
    args = parser.parse_args()

    args.ddi_file = args.ddi_file or args.data_dir / (
        "drugbank_ddi.tsv" if args.task == "multiclass" else "drugbank_ddibinary.txt"
    )
    args.entity_embedding_file = (
        args.entity_embedding_file or args.data_dir / "Drugbank_entity.npy"
    )
    args.graph_embedding_file = (
        args.graph_embedding_file or args.data_dir / "drugbank_mol_embeddings.npy"
    )
    args.protein_embedding_file = (
        args.protein_embedding_file or args.data_dir / "protein_embeddings.npy"
    )
    return args


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available.")
    return device


def make_loader(
    pairs: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(pairs).long(), torch.from_numpy(labels).long()
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


def evaluate(
    model: CMAFDDI,
    loader: DataLoader,
    device: torch.device,
    task: str,
    max_batches: int = 0,
) -> Dict[str, float]:
    model.eval()
    labels: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    scores: List[np.ndarray] = []

    with torch.inference_mode():
        for batch_number, (pairs, target) in enumerate(loader, start=1):
            pairs = pairs.to(device, non_blocking=True)
            logits = model(pairs)
            if task == "multiclass":
                prediction = logits.argmax(dim=1)
                score = torch.softmax(logits, dim=1)
            else:
                logits = logits.squeeze(1)
                score = torch.sigmoid(logits)
                prediction = (score >= 0.5).long()
            labels.append(target.numpy())
            predictions.append(prediction.cpu().numpy())
            scores.append(score.cpu().numpy())
            if max_batches and batch_number >= max_batches:
                break

    return classification_metrics(
        np.concatenate(labels),
        np.concatenate(predictions),
        np.concatenate(scores),
        task=task,
    )


def save_checkpoint(
    path: Path,
    model: CMAFDDI,
    args: argparse.Namespace,
    fold: int,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.config,
            "fold": fold,
            "epoch": epoch,
            "metrics": metrics,
            "data_files": {
                "ddi": str(args.ddi_file),
                "knowledge_graph": str(args.entity_embedding_file),
                "molecular_graph": str(args.graph_embedding_file),
                "protein_sequence": str(args.protein_embedding_file),
            },
        },
        path,
    )


def train_fold(
    args: argparse.Namespace,
    data: CMAFData,
    fold: int,
    device: torch.device,
) -> Dict[str, float]:
    train_idx, test_idx = data.folds[fold]
    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        data.pairs[train_idx],
        data.labels[train_idx],
        args.batch_size,
        shuffle=True,
        seed=args.seed + fold,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        data.pairs[test_idx],
        data.labels[test_idx],
        args.eval_batch_size,
        shuffle=False,
        seed=args.seed + fold,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = CMAFDDI(
        kg_embeddings=torch.from_numpy(data.kg_embeddings),
        molecular_embeddings=torch.from_numpy(data.molecular_embeddings),
        protein_embeddings=torch.from_numpy(data.protein_embeddings),
        output_dim=data.num_classes if args.task == "multiclass" else 1,
        fusion=args.fusion,
        attention_heads=args.attention_heads,
        ffp_dim=args.ffp_dim,
        hidden_dim_1=args.classifier_hidden_1,
        hidden_dim_2=args.classifier_hidden_2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion: nn.Module
    if args.task == "multiclass":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_metrics: Optional[Dict[str, float]] = None
    best_accuracy = -np.inf
    best_epoch = 0
    stale_evaluations = 0
    fold_dir = args.output_dir / f"fold_{fold}"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = perf_counter()
        total_loss = 0.0
        examples_seen = 0
        batches_seen = 0

        for batch_number, (pairs, target) in enumerate(train_loader, start=1):
            pairs = pairs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pairs)
            if args.task == "multiclass":
                loss = criterion(logits, target)
            else:
                loss = criterion(logits.squeeze(1), target.float())
            loss.backward()
            optimizer.step()

            batch_examples = pairs.shape[0]
            total_loss += loss.item() * batch_examples
            examples_seen += batch_examples
            batches_seen += 1
            if args.max_train_batches and batch_number >= args.max_train_batches:
                break

        mean_loss = total_loss / max(examples_seen, 1)
        LOGGER.info(
            "fold=%d epoch=%d loss=%.6f batches=%d seconds=%.2f",
            fold,
            epoch,
            mean_loss,
            batches_seen,
            perf_counter() - epoch_start,
        )

        if epoch % args.evaluate_every != 0:
            continue

        metrics = evaluate(
            model, test_loader, device, args.task, max_batches=args.max_eval_batches
        )
        LOGGER.info(
            "fold=%d epoch=%d macro_precision=%.4f macro_recall=%.4f "
            "macro_f1=%.4f accuracy=%.4f",
            fold,
            epoch,
            metrics["macro_precision"],
            metrics["macro_recall"],
            metrics["macro_f1"],
            metrics["accuracy"],
        )

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_metrics = dict(metrics)
            best_epoch = epoch
            stale_evaluations = 0
            if not args.no_save_checkpoints:
                save_checkpoint(
                    fold_dir / "best.pt", model, args, fold, epoch, metrics
                )
        else:
            stale_evaluations += 1

        if stale_evaluations >= args.patience:
            LOGGER.info("fold=%d early stopping at epoch=%d", fold, epoch)
            break

    if best_metrics is None:
        raise RuntimeError("No evaluation was run. Check --epochs and --evaluate-every.")
    best_metrics["fold"] = float(fold)
    best_metrics["best_epoch"] = float(best_epoch)
    best_metrics["train_examples"] = float(len(train_idx))
    best_metrics["test_examples"] = float(len(test_idx))
    return best_metrics


def serializable_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def write_results(
    output_dir: Path, args: argparse.Namespace, fold_metrics: List[Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = [
        key
        for key in fold_metrics[0]
        if key not in {"fold", "best_epoch", "train_examples", "test_examples"}
    ]
    summary = {
        name: {
            "mean": float(np.mean([row[name] for row in fold_metrics])),
            "std": float(np.std([row[name] for row in fold_metrics], ddof=0)),
        }
        for name in metric_names
    }

    with (output_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_metrics[0]))
        writer.writeheader()
        writer.writerows(fold_metrics)
    (output_dir / "run_config.json").write_text(
        json.dumps(serializable_args(args), indent=2), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(args.output_dir / "training.log")
    seed_everything(args.seed, deterministic=args.deterministic)
    device = select_device(args.device)
    LOGGER.info("device=%s", device)

    data = CMAFData.from_args(args)
    LOGGER.info("data=%s", data.describe())
    fold_count = len(data.folds)
    if args.max_folds:
        fold_count = min(fold_count, args.max_folds)

    fold_metrics = [
        train_fold(args, data, fold, device) for fold in range(fold_count)
    ]
    summary = write_results(args.output_dir, args, fold_metrics)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
