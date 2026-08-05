"""Data loading and alignment checks for CMAF-DDI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


Fold = Tuple[np.ndarray, np.ndarray]


@dataclass
class CMAFData:
    pairs: np.ndarray
    labels: np.ndarray
    kg_embeddings: np.ndarray
    molecular_embeddings: np.ndarray
    protein_embeddings: np.ndarray
    folds: List[Fold]
    num_classes: int

    @classmethod
    def from_args(cls, args: object) -> "CMAFData":
        pairs, labels = cls._load_ddi(Path(args.ddi_file))
        num_drugs = int(pairs.max()) + 1
        kg = cls._load_embeddings(
            Path(args.entity_embedding_file), "knowledge graph", args.entity_dim
        )
        molecular = cls._load_embeddings(
            Path(args.graph_embedding_file), "molecular graph", args.structure_dim
        )
        protein = cls._load_embeddings(
            Path(args.protein_embedding_file), "protein sequence", args.protein_dim
        )

        if kg.shape[0] < num_drugs:
            raise ValueError(
                f"Knowledge-graph embeddings contain {kg.shape[0]} rows, but DDI "
                f"indices require at least {num_drugs}."
            )
        kg = np.ascontiguousarray(kg[:num_drugs], dtype=np.float32)
        cls._require_exact_rows(molecular, num_drugs, "Molecular-graph")
        cls._require_exact_rows(protein, num_drugs, "Protein-sequence")

        unique_labels = np.unique(labels)
        if args.task == "multiclass":
            expected = np.arange(unique_labels.size)
            if not np.array_equal(unique_labels, expected):
                raise ValueError(
                    "Multiclass labels must be consecutive integers starting at 0; "
                    f"found {unique_labels.tolist()}."
                )
            num_classes = int(unique_labels.size)
            if num_classes < 2:
                raise ValueError("Multiclass training requires at least two labels.")
        else:
            if not set(unique_labels.tolist()).issubset({0, 1}):
                raise ValueError(
                    f"Binary labels must be 0 or 1; found {unique_labels.tolist()}."
                )
            num_classes = 2

        _, class_counts = np.unique(labels, return_counts=True)
        if class_counts.min() < args.folds:
            raise ValueError(
                f"The smallest class has {class_counts.min()} samples, fewer than "
                f"folds={args.folds}."
            )
        splitter = StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=args.split_seed
        )
        folds = [(train, test) for train, test in splitter.split(pairs, labels)]
        return cls(
            pairs=np.ascontiguousarray(pairs, dtype=np.int64),
            labels=np.ascontiguousarray(labels, dtype=np.int64),
            kg_embeddings=kg,
            molecular_embeddings=np.ascontiguousarray(molecular, dtype=np.float32),
            protein_embeddings=np.ascontiguousarray(protein, dtype=np.float32),
            folds=folds,
            num_classes=num_classes,
        )

    @staticmethod
    def _load_ddi(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(
                f"DDI file not found: {path}. See data/README.md for the layout."
            )
        frame = pd.read_csv(path, sep="\t", header=None)
        if frame.shape[1] != 3:
            raise ValueError(
                f"{path} must contain exactly three tab-separated columns; "
                f"found {frame.shape[1]}."
            )
        if frame.isna().any().any():
            raise ValueError(f"{path} contains missing values.")
        try:
            values = frame.to_numpy(dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} must contain integer values only.") from exc
        pairs = values[:, :2]
        labels = values[:, 2]
        if len(values) == 0:
            raise ValueError(f"{path} is empty.")
        if pairs.min() < 0:
            raise ValueError("Drug indices must be non-negative.")
        if labels.min() < 0:
            raise ValueError("Class labels must be non-negative.")
        return pairs, labels

    @staticmethod
    def _load_embeddings(path: Path, name: str, expected_dim: int) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(
                f"{name.title()} embedding file not found: {path}. "
                "See data/README.md for the layout."
            )
        matrix = np.load(path, allow_pickle=False)
        if matrix.ndim != 2:
            raise ValueError(f"{path} must be a 2-D matrix; found shape {matrix.shape}.")
        if matrix.shape[1] != expected_dim:
            raise ValueError(
                f"{name.title()} embeddings must have dimension {expected_dim}; "
                f"found shape {matrix.shape}."
            )
        if not np.issubdtype(matrix.dtype, np.number):
            raise ValueError(f"{path} must contain numeric values.")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{path} contains NaN or infinite values.")
        return matrix

    @staticmethod
    def _require_exact_rows(matrix: np.ndarray, expected: int, name: str) -> None:
        if matrix.shape[0] != expected:
            raise ValueError(
                f"{name} embeddings contain {matrix.shape[0]} rows, but DDI indices "
                f"require exactly {expected}. All modalities must use the same drug order."
            )

    def describe(self) -> Dict[str, object]:
        return {
            "ddi_samples": int(self.pairs.shape[0]),
            "drugs": int(self.molecular_embeddings.shape[0]),
            "classes": self.num_classes,
            "kg_shape": list(self.kg_embeddings.shape),
            "molecular_shape": list(self.molecular_embeddings.shape),
            "protein_shape": list(self.protein_embeddings.shape),
            "folds": len(self.folds),
        }
