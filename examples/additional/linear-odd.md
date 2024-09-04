# linear-odd

Evaluate regression with five features.

Mean-squared error gradients must be correct at an odd feature width.

Family: linear. Size: 5. Deterministic seed: 910106.

Run from the project directory:

~~~powershell
python -B examples/additional/run_cases.py --case linear-odd
~~~

Add --json to inspect returned metrics and output.

The adjacent expected file specifies the following metric conditions:

| Metric | Condition |
| --- | --- |
| elements | equals 10 |
| finite_gradients | equals true |
| graph_equivalent | equals true |
| graph_nodes_after | max 3 |
| max_gradient_error | min 0, max 0.0001 |

Scope: NumPy-backed differentiation and graph simplification.

These inputs are synthetic. See [project limitations](../../docs/LIMITATIONS.md).
