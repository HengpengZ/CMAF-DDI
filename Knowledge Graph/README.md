# Knowledge-graph provenance code

This directory contains the historical Keras KGNN implementation and raw files
from the research workspace. It is preserved for provenance, but it is **not**
called by the top-level `main.py` training path and has not been modernized as
the canonical paper preprocessing pipeline.

The paper-profile trainer consumes a precomputed 100-dimensional matrix through
`--entity-embedding-file`. See `data/README.md` for its required order, shape,
checksum, leakage-control requirement, and licensing boundary.

Important limitations of this legacy directory:

- configuration paths depend on the working directory;
- the code targets an old Keras API;
- `raw_data/DRKG/train2id.txt` contains 191,427 triples and must not be assumed
  to be the paper's verified 382,737-triple leakage-controlled DRKG snapshot;
- the top-level release does not claim that running `run.py` regenerates the
  manifest KG array.

Use these files only after independently verifying the source graph, entity
mapping, removal of direct benchmark DDI edges, software versions, and output
row order. A future KG preprocessing release should provide those steps as a
separate tested pipeline rather than silently changing this historical code.
