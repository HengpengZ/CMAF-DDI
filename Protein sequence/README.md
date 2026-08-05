# Protein sequence features

`run.py` generates one ESM2 embedding for each row in a drug-to-protein mapping
table. The default model is `facebook/esm2_t6_8M_UR50D`; masked mean pooling
over non-special residue tokens produces 320-dimensional vectors.

Required default columns are `DrugBank_ID`, `UniProt_ID`, and
`FASTA_Sequence`.

```bash
python "Protein sequence/run.py" \
  --input data/DRKG/drug_protein_sequences.csv \
  --output data/DRKG/protein_embeddings.npy \
  --batch-size 4
```

Rows missing a drug ID, UniProt ID, or sequence receive zero vectors. The
adjacent JSON file records counts, model identity, pooling, column names, and
maximum token length. Keep the input table in the canonical DDI drug order.
