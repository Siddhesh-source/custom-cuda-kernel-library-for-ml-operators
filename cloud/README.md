# Remote GPU execution: Kaggle

**All GPU work in this project runs on Kaggle**: builds, correctness tests,
compute-sanitizer, benchmarks and the PyTorch extension. Kaggle is free, needs
no payment method, and offers Tesla T4 (sm_75). The few components that cannot
run on Kaggle at all are listed in [`../docs/local.md`](../docs/local.md).

See [kaggle/README.md](kaggle/README.md) for credentials and usage:

    python cloud/kaggle/kaggle_run.py            # push, run on a T4, fetch the log

Results from Kaggle are filed under `results/sm_75/`. They are never mixed with
results from any other architecture.
