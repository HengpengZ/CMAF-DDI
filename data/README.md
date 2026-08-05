# Data contract

The paper-profile training entry consumes one DDI table and three precomputed
feature matrices. The matrices must use exactly the same drug-index order as
the DDI table.

## Required files

Place these files in `data/DRKG/` unless explicit paths are passed to
`main.py`.

| File | Type | Required content |
|---|---|---|
| `drugbank_ddi.tsv` | tab-separated integers, no header | 191,427 rows of `drug_1`, `drug_2`, `label` |
| `Drugbank_entity.npy` | float matrix | at least 1,706 rows by 100 KG features |
| `drugbank_mol_embeddings.npy` | float matrix | exactly 1,706 rows by 300 molecular features |
| `protein_embeddings.npy` | float matrix | exactly 1,706 rows by 320 protein features |

Drug and class indices are zero-based. For the paper profile, drug indices span
`0..1705` and labels span `0..85`. Labels must be consecutive.

The KG matrix may include non-drug entities after the first 1,706 rows. The
first rows must still follow the DDI drug order. Molecular and protein arrays
must have exactly one row per drug; padding or truncating without an explicit
mapping is not valid.

## Verified release-candidate identity

`artifact_manifest.json` records the filenames, shapes, and SHA-256 values of
the locally validated release-candidate set. Validate a prepared directory:

```bash
python scripts/validate_data.py --data-dir data/DRKG --paper-profile --hash
```

Shape and checksum validation confirms file identity and alignment constraints.
It does not replace a complete 200-epoch reproduction run.

## Source and licensing boundary

- **DrugBank:** obtain labels, structures, and drug mappings under a valid
  DrugBank license. DrugBank redistribution restrictions may apply to raw and
  derived files. This repository does not automate authentication or bypass
  those terms.
- **DRKG:** follow the DRKG project's data license and citation requirements.
- **UniProt:** follow UniProt's terms for protein sequences and mappings.
- **ESM2 and DGL-LifeSci:** their model weights and software retain their own
  licenses.

Before attaching arrays to a GitHub Release or an archival repository, review
all four sources and document the exact preprocessing provenance. If DrugBank
redistribution is not permitted, publish scripts plus checksums and require
users to construct the licensed inputs locally.

## Alignment checklist

1. Freeze a table mapping `drug_index` to DrugBank ID.
2. Generate KG, molecular, and protein rows from that table without re-sorting.
3. Keep zero rows for missing modality values and record them in metadata.
4. Remove any DDI samples whose drug indices are outside the feature matrices.
5. Remove direct benchmark DDI triples from the KG feature-training graph.
6. Run the validator and archive its JSON report with the experiment.

The files currently under `Knowledge Graph/raw_data/DRKG/` are retained from
the historical codebase. They are not used by `main.py` and must not be treated
as a verified leakage-controlled reconstruction of the paper features.
