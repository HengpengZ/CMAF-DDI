# Molecular graph features

`pretrain_smiles_embedding.py` converts a row-ordered SMILES table into a NumPy
matrix using a pretrained DGL-LifeSci GIN and average graph pooling.

```bash
python "Molecular Graph/pretrain_smiles_embedding.py" \
  --input data/DRKG/drug_smiles.tsv \
  --delimiter "\t" \
  --smiles-column SMILES \
  --model gin_supervised_masking \
  --output data/DRKG/drugbank_mol_embeddings.npy
```

The output retains every input row. Invalid or missing SMILES receive zero
vectors and are listed in the adjacent JSON metadata file. Review those rows
before training; the paper profile expects exactly 1,706 rows by 300 features.
