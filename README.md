# Deep Learning Graph Optimization Framework

A NumPy reverse-mode autodiff framework with independently checked gradients and a small pure-expression optimizer.

This is newly generated educational reference code based on a concept in the supplied PDF.
It has **165 non-empty source, test, configuration, workload and documentation files**.
It is a prototype for study and extension, not evidence of previous deployment or measured large-scale performance.

## Quick start

Requires Python 3.10+ and NumPy (`python -m pip install -r requirements.txt`).

```sh
python scripts/demo.py
python scripts/run_tests.py
python scripts/benchmark.py
python scripts/serve.py --port 8080
```

Open http://127.0.0.1:8080 for the workload dashboard. `scripts/demo.py --case workloads/<id>.case.json`
executes one workload and checks its independent acceptance conditions. `benchmark.py` prints actual local timings.

## Implemented scope

Tensor graph with reverse-mode autodiff, scalar/broadcast arithmetic, 2D matrix multiplication, reduction, slicing, stable sigmoid and cross entropy, Linear/MLP modules, SGD and constant/identity folding.

## Limits and optional runtimes

No GPU backend, custom native matmul or automatic tensor-graph fusion. The expression optimizer is separate from the dynamic Tensor graph. Backward resets all reachable gradients per invocation. Matrix multiplication is 2D only; sum supports one axis. Recursive traversal is unsuitable for extremely deep graphs.

All bundled data are synthetic. No credentials, pretrained model weights, historical commits, or fabricated benchmark results are included.
See `docs/PROVENANCE.md`, `docs/TESTING.md`, and the bundle's validation report for evidence and omissions.
