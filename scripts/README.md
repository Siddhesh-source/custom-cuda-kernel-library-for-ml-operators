# scripts

| Script | Purpose |
|---|---|
| `setup_env.sh` | Checks the tool versions (`nvidia-smi`, `nvcc`, `ncu`, `nsys`, `compute-sanitizer`) and creates a venv on top of the CUDA PyTorch (reference.md "Common setup") |
| `sanitize.sh` | compute-sanitizer memcheck, racecheck, initcheck and synccheck over every test suite; writes a summary table |
| `make_shapes.py` | Generates `bench/shapes.json`, the single source of truth for benchmark shapes |
| `benchlib.py` | Python mirror of `bench/support/`: statistics, gate, telemetry, CSV |
| `bench_torch.py` | PyTorch eager, torch.compile and Triton baselines, using the same method and schema as the C++ benches |
| `triton_kernels.py` | The Triton baseline kernels (softmax, layernorm, matmul) |
| `kaggle_job.sh` | What one Kaggle GPU session runs after the build: env, tests, sanitizer, peaks, all benches |
| `local_build.ps1` | Local sm_89 build, for the items in docs/local.md only |
| `tests/` | pytest for `benchlib.py` |

See [docs/benchmarking.md](../docs/benchmarking.md) for the measurement method.
