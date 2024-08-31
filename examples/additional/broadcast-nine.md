# broadcast-nine

Broadcast a nine-column bias over two rows.

The wider reduction must preserve gradient correctness.

Family: broadcast. Size: 9. Deterministic seed: 910108.

Run from the project directory:

~~~powershell
python -B examples/additional/run_cases.py --case broadcast-nine
~~~

Add --json to inspect returned metrics and output.

The adjacent expected file specifies the following metric conditions:

| Metric | Condition |
| --- | --- |
| elements | equals 18 |
| finite_gradients | equals true |
| graph_equivalent | equals true |
| graph_nodes_after | max 3 |
| max_gradient_error | min 0, max 0.0001 |

Scope: NumPy-backed differentiation and graph simplification.

These inputs are synthetic. See [project limitations](../../docs/LIMITATIONS.md).
