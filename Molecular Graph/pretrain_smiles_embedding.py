"""Generate row-aligned molecular graph embeddings from SMILES."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import (
    PretrainAtomFeaturizer,
    PretrainBondFeaturizer,
    mol_to_bigraph,
)
from rdkit import Chem
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV/TSV SMILES file.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy file.")
    parser.add_argument("--delimiter", default="\t")
    parser.add_argument("--smiles-column", default="SMILES")
    parser.add_argument(
        "--model",
        choices=(
            "gin_supervised_contextpred",
            "gin_supervised_infomax",
            "gin_supervised_edgepred",
            "gin_supervised_masking",
        ),
        default="gin_supervised_masking",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_graphs(smiles: Sequence[object]) -> Tuple[List[dgl.DGLGraph], List[int], List[int]]:
    graphs: List[dgl.DGLGraph] = []
    row_ids: List[int] = []
    invalid_rows: List[int] = []
    for row_id, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(str(value)) if pd.notna(value) else None
        if molecule is None:
            invalid_rows.append(row_id)
            continue
        graph = mol_to_bigraph(
            molecule,
            add_self_loop=True,
            node_featurizer=PretrainAtomFeaturizer(),
            edge_featurizer=PretrainBondFeaturizer(),
            canonical_atom_order=False,
        )
        graphs.append(graph)
        row_ids.append(row_id)
    return graphs, row_ids, invalid_rows


def collate(samples: Sequence[Tuple[dgl.DGLGraph, int]]) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    graphs, row_ids = zip(*samples)
    return dgl.batch(graphs), torch.tensor(row_ids, dtype=torch.long)


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input, sep=args.delimiter)
    if args.smiles_column not in frame.columns:
        raise ValueError(
            f"Column {args.smiles_column!r} was not found in {args.input}."
        )
    graphs, row_ids, invalid_rows = build_graphs(frame[args.smiles_column].tolist())
    if not graphs:
        raise ValueError("No valid SMILES strings were found.")

    device = select_device(args.device)
    model = load_pretrained(args.model).to(device)
    model.eval()
    readout = AvgPooling()
    loader = DataLoader(
        list(zip(graphs, row_ids)),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    output = None
    for batch_number, (graph, batch_rows) in enumerate(loader, start=1):
        graph = graph.to(device)
        node_features = [
            graph.ndata.pop("atomic_number"),
            graph.ndata.pop("chirality_type"),
        ]
        edge_features = [
            graph.edata.pop("bond_type"),
            graph.edata.pop("bond_direction_type"),
        ]
        with torch.inference_mode():
            node_representations = model(graph, node_features, edge_features)
            batch_embeddings = readout(graph, node_representations).cpu().numpy()
        if output is None:
            output = np.zeros(
                (len(frame), batch_embeddings.shape[1]), dtype=np.float32
            )
        output[batch_rows.numpy()] = batch_embeddings
        print(f"Processed batch {batch_number}/{len(loader)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "model": args.model,
        "pooling": "average graph readout",
        "rows": len(frame),
        "valid_rows": len(row_ids),
        "zero_rows": len(invalid_rows),
        "invalid_row_indices": invalid_rows,
        "embedding_dim": int(output.shape[1]),
        "smiles_column": args.smiles_column,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
