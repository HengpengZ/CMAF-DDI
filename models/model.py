"""CMAF-DDI model components."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class CMAFDDI(nn.Module):
    """Fuse drug modalities and classify directed drug pairs.

    The default dimensions reproduce the architecture described in the paper:
    100-D KG, 300-D molecular, and 320-D protein representations; a 100-D
    triple-feature product; and a 2,460-D Feature Fusion Perception output.
    """

    def __init__(
        self,
        kg_embeddings: torch.Tensor,
        molecular_embeddings: torch.Tensor,
        protein_embeddings: torch.Tensor,
        output_dim: int,
        fusion: str = "cmaf",
        attention_heads: int = 4,
        ffp_dim: int = 2460,
        hidden_dim_1: int = 2048,
        hidden_dim_2: int = 2048,
    ) -> None:
        super().__init__()
        self._validate_embeddings(
            kg_embeddings, molecular_embeddings, protein_embeddings
        )
        self.fusion = fusion.lower()
        if self.fusion not in {"cmaf", "concat", "sum"}:
            raise ValueError(f"Unsupported fusion method: {fusion}")

        self.num_drugs = molecular_embeddings.shape[0]
        self.kg_dim = kg_embeddings.shape[1]
        self.molecular_dim = molecular_embeddings.shape[1]
        self.protein_dim = protein_embeddings.shape[1]
        self.raw_dim = self.kg_dim + self.molecular_dim + self.protein_dim

        # Features are fixed inputs and deliberately omitted from checkpoints.
        self.register_buffer("kg_embeddings", kg_embeddings.float(), persistent=False)
        self.register_buffer(
            "molecular_embeddings", molecular_embeddings.float(), persistent=False
        )
        self.register_buffer(
            "protein_embeddings", protein_embeddings.float(), persistent=False
        )

        if self.fusion == "cmaf":
            if self.raw_dim % attention_heads != 0:
                raise ValueError(
                    f"Concatenated dimension {self.raw_dim} must be divisible by "
                    f"attention_heads={attention_heads}."
                )
            self.molecular_projection = nn.Linear(self.molecular_dim, self.kg_dim)
            self.kg_projection = nn.Linear(self.kg_dim, self.kg_dim)
            self.protein_projection = nn.Linear(self.protein_dim, self.kg_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=self.raw_dim, num_heads=attention_heads
            )
            self.triple_product_mlp = nn.Sequential(
                nn.Linear(self.kg_dim, self.kg_dim), nn.ReLU()
            )
            self.ffp = nn.Sequential(
                nn.Linear(self.raw_dim + self.kg_dim, ffp_dim),
                nn.ReLU(),
                nn.Linear(ffp_dim, ffp_dim),
            )
            drug_embedding_dim = ffp_dim
        elif self.fusion == "concat":
            self.concat_fusion = nn.Sequential(
                nn.Linear(self.raw_dim, self.kg_dim),
                nn.BatchNorm1d(self.kg_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.kg_dim, self.kg_dim),
                nn.BatchNorm1d(self.kg_dim),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.kg_dim, self.kg_dim),
                nn.BatchNorm1d(self.kg_dim),
                nn.LeakyReLU(inplace=True),
            )
            drug_embedding_dim = self.kg_dim
        else:
            self.molecular_projection = nn.Linear(self.molecular_dim, self.kg_dim)
            self.kg_projection = nn.Linear(self.kg_dim, self.kg_dim)
            self.protein_projection = nn.Linear(self.protein_dim, self.kg_dim)
            drug_embedding_dim = self.kg_dim

        self.classifier = nn.Sequential(
            nn.Linear(2 * drug_embedding_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_2, output_dim),
        )
        self.config: Dict[str, object] = {
            "output_dim": output_dim,
            "fusion": self.fusion,
            "attention_heads": attention_heads,
            "ffp_dim": ffp_dim,
            "hidden_dim_1": hidden_dim_1,
            "hidden_dim_2": hidden_dim_2,
            "kg_dim": self.kg_dim,
            "molecular_dim": self.molecular_dim,
            "protein_dim": self.protein_dim,
            "num_drugs": self.num_drugs,
        }

    @staticmethod
    def _validate_embeddings(
        kg_embeddings: torch.Tensor,
        molecular_embeddings: torch.Tensor,
        protein_embeddings: torch.Tensor,
    ) -> None:
        matrices = {
            "knowledge graph": kg_embeddings,
            "molecular graph": molecular_embeddings,
            "protein sequence": protein_embeddings,
        }
        for name, matrix in matrices.items():
            if matrix.ndim != 2:
                raise ValueError(f"{name} embeddings must be a 2-D matrix.")
        rows = {name: matrix.shape[0] for name, matrix in matrices.items()}
        if len(set(rows.values())) != 1:
            raise ValueError(f"Embedding row counts are not aligned: {rows}")

    def encode_drugs(self) -> torch.Tensor:
        raw = torch.cat(
            [self.kg_embeddings, self.molecular_embeddings, self.protein_embeddings],
            dim=1,
        )
        if self.fusion == "concat":
            return self.concat_fusion(raw)
        if self.fusion == "sum":
            return (
                self.kg_projection(self.kg_embeddings)
                + self.molecular_projection(self.molecular_embeddings)
                + self.protein_projection(self.protein_embeddings)
            )

        projected_product = (
            self.kg_projection(self.kg_embeddings)
            * self.molecular_projection(self.molecular_embeddings)
            * self.protein_projection(self.protein_embeddings)
        )
        triple_feature = self.triple_product_mlp(projected_product)

        # This layout preserves the released implementation: drugs form the
        # attention batch and the concatenated feature vector forms one token.
        attention_output, _ = self.attention(
            raw.unsqueeze(0), raw.unsqueeze(0), raw.unsqueeze(0), need_weights=False
        )
        fused = torch.cat([attention_output.squeeze(0), triple_feature], dim=1)
        return self.ffp(fused)

    def forward(self, drug_pairs: torch.Tensor) -> torch.Tensor:
        if drug_pairs.ndim != 2 or drug_pairs.shape[1] != 2:
            raise ValueError("drug_pairs must have shape [batch_size, 2].")
        if drug_pairs.dtype != torch.long:
            drug_pairs = drug_pairs.long()
        if drug_pairs.numel() and (
            drug_pairs.min().item() < 0
            or drug_pairs.max().item() >= self.num_drugs
        ):
            raise IndexError(
                f"Drug indices must be between 0 and {self.num_drugs - 1}."
            )

        drug_embeddings = self.encode_drugs()
        pair_embeddings = torch.cat(
            [
                drug_embeddings[drug_pairs[:, 0]],
                drug_embeddings[drug_pairs[:, 1]],
            ],
            dim=1,
        )
        return self.classifier(pair_embeddings)
