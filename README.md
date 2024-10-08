# Deep Learning Graph Optimization Framework

A NumPy reverse-mode autodiff framework with independently checked gradients and a small pure-expression optimizer.

## 1. Overview

Automatic differentiation is easier to understand when the computation graph and gradient checks are visible. This NumPy reference implements reverse-mode differentiation alongside a separate pure-expression simplifier, so numerical correctness and graph equivalence can be studied independently.

**Project type:** educational reference implementation. **Repository contents:** 165 non-empty source, test, configuration, workload and documentation files, including 36 synthetic workload scenarios.

## 2. Core Features — Why They Matter

- **Reverse-mode autodiff:** Builds a dynamic tensor graph and propagates gradients backward.
- **Tensor operators:** Supports arithmetic, broadcasting, 2D matrix multiplication, reductions, slicing and stable loss operations.
- **Small neural modules:** Includes Linear/MLP building blocks and SGD.
- **Independent correctness checks:** Compares gradients with central differences and tests simplifier equivalence.

## 3. Tech Stack & Architecture

| Layer | Technology | Implementation status |
| --- | --- | --- |
| Execution | Python 3.10+, NumPy >=1.24,<3 | Runnable CPU implementation |
| Differentiation | Custom Tensor graph and reverse traversal | Implemented locally |
| Optimization | Pure-expression AST constant/identity folding | Separate from the dynamic Tensor graph |

### How the components fit together

Tensor operations create a dynamic directed acyclic graph. Reverse traversal accumulates parameter gradients, while numerical differentiation supplies an independent oracle. A separate expression AST is simplified and evaluated before and after transformation.

| Component | Responsibility |
| --- | --- |
| [src/syslab/tensor.py](src/syslab/tensor.py) | Tensor operations, graph edges and backward propagation. |
| [src/syslab/operators.py](src/syslab/operators.py) | Loss operators and numerical gradient checking. |
| [src/syslab/graph.py](src/syslab/graph.py) | Expression simplification, evaluation and node counting. |
| [src/syslab/nn.py](src/syslab/nn.py) | Linear and MLP modules. |

See [Architecture](docs/ARCHITECTURE.md) and [Algorithms](docs/ALGORITHMS.md) for implementation notes.

### Scope and limitations

No GPU backend, custom native matmul or automatic tensor-graph fusion. The expression optimizer is separate from the dynamic Tensor graph. Backward resets all reachable gradients per invocation. Matrix multiplication is 2D only; sum supports one axis. Recursive traversal is unsuitable for extremely deep graphs.

## 4. Getting Started / Installation

**Prerequisites:** Python 3.10+ and NumPy (`numpy>=1.24,<3`). No API keys or external services are needed for the default sample.

Download/extract this project or clone its repository, then open a terminal in the `autograd-graph-optimizer` folder. Create an isolated environment:

```sh
python -m venv .venv
```

Activate it on Linux/macOS:

```sh
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the declared Python dependencies and run the demo:

```sh
python -m pip install -r requirements.txt
python scripts/demo.py
```

Check behavior and collect timings on your own machine:

```sh
python scripts/run_tests.py
python scripts/benchmark.py
```

The original bundle validation recorded **30 passing tests** for this project and **36 accepted workload scenarios**. These are local reference checks, not production or hardware benchmarks. See [Testing](docs/TESTING.md) and [Benchmark notes](docs/BENCHMARKS.md).

## 5. Usage Examples

### Run a reproducible workload

The bundled [sample request](examples/request.json) contains:

```json
{
  "family": "scalar",
  "id": "scalar-002-01",
  "seed": 101,
  "size": 2
}
```

Run the corresponding workload and check its independent acceptance conditions:

```sh
python scripts/demo.py --case workloads/scalar-002-01.case.json
```

Expected `metrics` excerpt from the verified local run; the complete JSON also includes `output`:

```json
{
  "metrics": {
    "elements": 4,
    "finite_gradients": true,
    "graph_equivalent": true,
    "graph_nodes_after": 3,
    "graph_nodes_before": 7,
    "loss": 0.681906439897148,
    "max_gradient_error": 5.99054139627242e-12
  }
}
```

The scalar example checks four input elements. Its maximum gradient error is approximately 6e-12, and the separate expression graph shrinks from seven nodes to three with equivalent evaluation. This is a small CPU correctness example, not a training-speed result.

The complete example is in [examples/response.json](examples/response.json). Floating-point last digits can vary across numeric environments.

### Explore through the local dashboard

```sh
python scripts/serve.py --port 8080
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080), select a workload and choose **Run and check**. The server listens on loopback and is intended for local inspection.

With the server running, a second terminal can call its workload inspection API:

```sh
curl "http://127.0.0.1:8080/api/run?id=scalar-002-01"
```

This endpoint executes the bundled workload; it is not a production domain API.

## 6. Your Contributions / Research Alignment

### Implementation evidence

The following work areas are present in this reference and can be reviewed directly:

| Work area in this reference | Repository evidence |
| --- | --- |
| Reverse-mode autodiff | [src/syslab/tensor.py](src/syslab/tensor.py) |
| Correctness and edge cases | [tests/](tests/) and [acceptance workloads](workloads/) |
| Reproducible evaluation | [scripts/demo.py](scripts/demo.py), [scripts/benchmark.py](scripts/benchmark.py), [testing notes](docs/TESTING.md) |

### Research alignment

The project connects deep-learning foundations with numerical analysis and compiler transformations. It provides concrete material for discussing gradient correctness, operator semantics and the boundary between graph optimization and execution.

**A question to investigate:** Which algebraic rewrites preserve outputs and gradients, and when do their numerical or runtime effects differ?

This question is a proposed extension, not a completed research result. Evaluate it with controlled inputs, independent correctness checks and measurements tied to a reproducible configuration.

### Personal contribution record

This reference was generated from the supplied project concept. Personal authorship or research contributions have not been verified. For an MS application, document only the modules you actually changed, the design choices you can explain, and experiments you ran; link those claims to commits or reproducible reports. See [Provenance](docs/PROVENANCE.md).

All bundled inputs are synthetic. The repository does not establish historical development dates, prior deployment or published research.
