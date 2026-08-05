# Reproducibility protocol

This document separates three claims that are often conflated:

1. **Code smoke test:** imports, data contracts, forward pass, backward pass,
   and architecture parameter count work.
2. **Artifact validation:** the four inputs have the expected identities,
   dimensions, labels, and row counts.
3. **Paper reproduction:** the complete five-fold, 200-epoch experiment is run
   from the validated artifacts and its fold metrics are compared with the
   accepted manuscript.

CI covers the first claim. `scripts/validate_data.py` covers the second. A full
GPU run is required for the third.

## 1. Record the environment

Create the core environment and capture its state before training:

```bash
conda env create -f environment.yml
conda activate cmaf-ddi
python --version
python -c "import numpy, pandas, sklearn, torch; print(numpy.__version__, pandas.__version__, sklearn.__version__, torch.__version__)"
```

For GPU runs, also archive `nvidia-smi` output. The code was smoke-tested during
release preparation with Python 3.8.19, NumPy 1.24.4, pandas 2.0.3,
scikit-learn 1.3.2, and PyTorch 2.4.1 on CPU. The historical experiment
environment used PyTorch 1.10.2 and CUDA 11.3; numerical differences across
hardware and libraries remain possible.

## 2. Validate the input snapshot

```bash
python scripts/validate_data.py \
  --data-dir data/DRKG \
  --paper-profile \
  --hash \
  --output outputs/data_validation.json
```

The expected paper-profile counts are:

- 191,427 directed DDI samples;
- 1,706 drugs;
- 86 relation types;
- aligned feature widths of 100, 300, and 320.

Compare hashes with `data/artifact_manifest.json`. A mismatch is not always an
error, but it means the artifact snapshot differs and must be documented.

## 3. Run the paper configuration

```bash
python main.py \
  --data-dir data/DRKG \
  --task multiclass \
  --fusion cmaf \
  --seed 2020 \
  --split-seed 3 \
  --folds 5 \
  --epochs 200 \
  --learning-rate 0.0001 \
  --batch-size 2048 \
  --eval-batch-size 8192 \
  --attention-heads 4 \
  --ffp-dim 2460 \
  --deterministic \
  --device cuda \
  --output-dir outputs/paper-cmaf
```

The run uses stratified fold splits and computes metrics once over all
predictions in each held-out fold. Evaluation loaders are not shuffled.
Early stopping monitors held-out accuracy with patience 10, matching the main
selection criterion in the released implementation.

The code creates one checkpoint directory per fold, avoiding the historical
behavior where folds could overwrite identically named checkpoints.

## 4. Run fusion ablations

Use the same inputs and seeds, changing only `--fusion`:

```bash
python main.py --data-dir data/DRKG --fusion concat --output-dir outputs/concat
python main.py --data-dir data/DRKG --fusion sum --output-dir outputs/sum
```

Modality-removal, AF-removal, TPF-removal, perturbation, UMAP, and case-study
analyses reported in the paper require dedicated experiment code and saved
fold checkpoints. They are not represented by changing an undocumented flag
in this release. Do not claim those tables were reproduced unless the exact
analysis pipeline and outputs are archived.

## 5. Interpret outputs

`fold_metrics.csv` contains the best held-out metrics and epoch for every fold.
`summary.json` stores the population mean and standard deviation across folds.
`run_config.json` records all CLI values. `training.log` is the chronological
trace. Each `fold_<n>/best.pt` file contains weights, model configuration,
epoch, metrics, and input paths, but not the large fixed feature matrices.

For an archival experiment, save together:

- Git commit SHA;
- environment export and GPU information;
- data-validation JSON and artifact hashes;
- all five fold checkpoints;
- `fold_metrics.csv`, `summary.json`, `run_config.json`, and `training.log`.

## Known boundaries

- DrugBank licensing may prevent public redistribution of the label table or
  derived mappings.
- The historical `Knowledge Graph/` Keras code is provenance material and is
  not invoked by the paper-profile training entry.
- The locally discovered 1,702-row molecular array is invalid for the 1,706
  drug profile. The manifest identifies the corrected 1,706-row candidate.
- The accepted paper's final DOI was not present in the supplied manuscript at
  release-preparation time.
