"""Validate CMAF-DDI input files and optionally compute checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import CMAFData  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/DRKG"))
    parser.add_argument("--ddi-file", type=Path, default=None)
    parser.add_argument("--entity-embedding-file", type=Path, default=None)
    parser.add_argument("--graph-embedding-file", type=Path, default=None)
    parser.add_argument("--protein-embedding-file", type=Path, default=None)
    parser.add_argument("--task", choices=("multiclass", "binary"), default="multiclass")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=3)
    parser.add_argument("--paper-profile", action="store_true")
    parser.add_argument("--hash", action="store_true", dest="compute_hash")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    files = {
        "ddi": args.ddi_file
        or args.data_dir
        / ("drugbank_ddi.tsv" if args.task == "multiclass" else "drugbank_ddibinary.txt"),
        "knowledge_graph": args.entity_embedding_file
        or args.data_dir / "Drugbank_entity.npy",
        "molecular_graph": args.graph_embedding_file
        or args.data_dir / "drugbank_mol_embeddings.npy",
        "protein_sequence": args.protein_embedding_file
        or args.data_dir / "protein_embeddings.npy",
    }
    loader_args = SimpleNamespace(
        ddi_file=files["ddi"],
        entity_embedding_file=files["knowledge_graph"],
        graph_embedding_file=files["molecular_graph"],
        protein_embedding_file=files["protein_sequence"],
        entity_dim=100,
        structure_dim=300,
        protein_dim=320,
        task=args.task,
        folds=args.folds,
        split_seed=args.split_seed,
    )
    data = CMAFData.from_args(loader_args)
    report: Dict[str, object] = {"status": "ok", **data.describe()}
    report["files"] = {name: str(path) for name, path in files.items()}
    if args.compute_hash:
        report["sha256"] = {name: sha256(path) for name, path in files.items()}
    if args.paper_profile:
        expected = {"ddi_samples": 191427, "drugs": 1706, "classes": 86}
        mismatches = {
            key: {"expected": value, "actual": report[key]}
            for key, value in expected.items()
            if report[key] != value
        }
        if mismatches:
            report["status"] = "error"
            report["paper_profile_mismatches"] = mismatches

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
