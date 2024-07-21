# scalar-003-02

Differentiate a quadratic expression and compare central differences.

Input scale: 3; deterministic random seed: 102.
Run `python scripts/demo.py --case workloads/scalar-003-02.case.json`.
The adjacent expected file specifies acceptance conditions independently from the implementation.
A successful run validates these conditions, not a production latency or capacity guarantee.

Inspect `metrics` and `output` in the returned JSON. Increase the input scale only after reading
`docs/LIMITATIONS.md`; do not extrapolate synthetic timing to hardware or external services.
