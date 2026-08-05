"""Run a dependency-light data, forward, and backward smoke test."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import CMAFDDI  # noqa: E402
from utils import CMAFData  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory(prefix="cmafddi-smoke-") as directory:
        root = Path(directory)
        labels = np.tile(np.arange(3), 20)
        pairs = rng.integers(0, 12, size=(labels.size, 2), dtype=np.int64)
        np.savetxt(
            root / "drugbank_ddi.tsv",
            np.column_stack([pairs, labels]),
            delimiter="\t",
            fmt="%d",
        )
        np.save(root / "kg.npy", rng.normal(size=(20, 100)).astype(np.float32))
        np.save(root / "mol.npy", rng.normal(size=(12, 300)).astype(np.float32))
        np.save(root / "protein.npy", rng.normal(size=(12, 320)).astype(np.float32))

        args = SimpleNamespace(
            ddi_file=root / "drugbank_ddi.tsv",
            entity_embedding_file=root / "kg.npy",
            graph_embedding_file=root / "mol.npy",
            protein_embedding_file=root / "protein.npy",
            entity_dim=100,
            structure_dim=300,
            protein_dim=320,
            task="multiclass",
            folds=3,
            split_seed=3,
        )
        data = CMAFData.from_args(args)
        model = CMAFDDI(
            torch.from_numpy(data.kg_embeddings),
            torch.from_numpy(data.molecular_embeddings),
            torch.from_numpy(data.protein_embeddings),
            output_dim=3,
            ffp_dim=32,
            hidden_dim_1=32,
            hidden_dim_2=16,
        )
        batch_pairs = torch.from_numpy(data.pairs[:8]).long()
        batch_labels = torch.from_numpy(data.labels[:8]).long()
        logits = model(batch_pairs)
        loss = torch.nn.functional.cross_entropy(logits, batch_labels)
        loss.backward()
        if logits.shape != (8, 3) or not torch.isfinite(loss):
            raise RuntimeError("CMAF-DDI smoke test produced invalid output.")

        reference = CMAFDDI(
            torch.zeros(4, 100),
            torch.zeros(4, 300),
            torch.zeros(4, 320),
            output_dim=86,
        )
        parameter_count = sum(parameter.numel() for parameter in reference.parameters())
        if parameter_count != 24_691_566:
            raise RuntimeError(f"Unexpected paper-model parameter count: {parameter_count}")

    print(
        "Smoke test passed: aligned data, forward/backward pass, "
        f"and paper parameter count ({parameter_count:,})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
