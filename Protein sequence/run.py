"""Generate drug-aligned ESM2 protein-sequence embeddings."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV/TSV mapping file.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy file.")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--drug-column", default="DrugBank_ID")
    parser.add_argument("--protein-column", default="UniProt_ID")
    parser.add_argument("--sequence-column", default="FASTA_Sequence")
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def valid_sequence(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.input, sep=args.delimiter)
    required = {args.drug_column, args.protein_column, args.sequence_column}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    device = select_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    model.eval()
    embedding_dim = int(model.config.hidden_size)
    model_limit = int(getattr(model.config, "max_position_embeddings", 1026))
    max_length = args.max_length or model_limit

    embeddings = np.zeros((len(frame), embedding_dim), dtype=np.float32)
    valid_rows: List[int] = [
        index
        for index, row in frame.iterrows()
        if pd.notna(row[args.drug_column])
        and pd.notna(row[args.protein_column])
        and valid_sequence(row[args.sequence_column])
    ]
    special_ids = set(tokenizer.all_special_ids)

    for start in range(0, len(valid_rows), args.batch_size):
        row_ids = valid_rows[start : start + args.batch_size]
        sequences = [str(frame.at[index, args.sequence_column]).strip() for index in row_ids]
        encoded = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        amp_context = (
            torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()
        )
        with torch.inference_mode(), amp_context:
            hidden = model(**encoded).last_hidden_state

        valid_mask = encoded["attention_mask"].bool()
        for token_id in special_ids:
            valid_mask &= encoded["input_ids"] != token_id
        denominator = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (hidden * valid_mask.unsqueeze(-1)).sum(dim=1) / denominator
        embeddings[row_ids] = pooled.float().cpu().numpy()
        print(f"Processed {min(start + args.batch_size, len(valid_rows))}/{len(valid_rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "model": args.model,
        "pooling": "masked mean over non-special residue tokens",
        "rows": len(frame),
        "valid_rows": len(valid_rows),
        "zero_rows": len(frame) - len(valid_rows),
        "embedding_dim": embedding_dim,
        "max_length": max_length,
        "drug_column": args.drug_column,
        "protein_column": args.protein_column,
        "sequence_column": args.sequence_column,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
