# matmul-eleven

Differentiate matrix multiplication with eleven features.

An odd inner dimension checks the backward matrix products.

Family: matmul. Size: 11. Deterministic seed: 910109.

Run from the project directory:

~~~powershell
python -B examples/additional/run_cases.py --case matmul-eleven
~~~

Add --json to inspect returned metrics and output.

The adjacent expected file specifies the following metric conditions:

| Metric | Condition |
| --- | --- |
| elements | equals 22 |
| finite_gradients | equals true |
| graph_equivalent | equals true |
| graph_nodes_after | max 3 |
| max_gradient_error | min 0, max 0.0001 |

Scope: NumPy-backed differentiation and graph simplification.

These inputs are synthetic. See [project limitations](../../docs/LIMITATIONS.md).
