# Contributing

Contributions that improve reproducibility, portability, tests, or
documentation are welcome.

1. Create a focused branch and keep unrelated formatting changes out of the
   patch.
2. Run `python scripts/smoke_test.py` before opening a pull request.
3. For data-pipeline changes, also run `python scripts/validate_data.py` on a
   legally obtained local dataset.
4. Document any change to drug ordering, split construction, feature models,
   dimensions, or metrics. These changes can invalidate comparisons with the
   paper.
5. Do not commit DrugBank data, credentials, local paths, model checkpoints, or
   generated embeddings without confirming the applicable license and release
   policy.

Bug reports should include the command, platform, Python/PyTorch versions, full
traceback, input shapes, and whether the smoke test passes. Do not attach
licensed datasets to public issues.
