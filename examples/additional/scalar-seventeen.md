# scalar-seventeen

Differentiate a quadratic over thirty-four elements.

The mean must scale gradients by the full element count.

Family: scalar. Size: 17. Deterministic seed: 910110.

Run from the project directory:

~~~powershell
python -B examples/additional/run_cases.py --case scalar-seventeen
~~~

Add --json to inspect returned metrics and output.

The adjacent expected file specifies the following metric conditions:

| Metric | Condition |
| --- | --- |
| elements | equals 34 |
| finite_gradients | equals true |
| graph_equivalent | equals true |
| graph_nodes_after | max 3 |
| max_gradient_error | min 0, max 0.0001 |

Scope: NumPy-backed differentiation and graph simplification.

These inputs are synthetic. See [project limitations](../../docs/LIMITATIONS.md).
