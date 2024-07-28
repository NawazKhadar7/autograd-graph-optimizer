# broadcast-012-06

Reduce broadcast gradients back to the original operand shape.

Input scale: 12; deterministic random seed: 137.
Run `python scripts/demo.py --case workloads/broadcast-012-06.case.json`.
The adjacent expected file specifies acceptance conditions independently from the implementation.
A successful run validates these conditions, not a production latency or capacity guarantee.

Inspect `metrics` and `output` in the returned JSON. Increase the input scale only after reading
`docs/LIMITATIONS.md`; do not extrapolate synthetic timing to hardware or external services.
