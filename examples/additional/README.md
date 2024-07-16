# Additional scenarios for autograd-graph-optimizer

Ten runnable scenarios cover small inputs, odd sizes, and behavior boundaries in the existing reference implementation.

This directory contains 10 input JSON files, 10 metric-oracle JSON files, 10 scenario notes, this guide, and the runner (32 files).

Run from the project directory with Python 3.10 or later and the dependencies already listed in requirements.txt.

~~~powershell
python -B examples/additional/run_cases.py --list
python -B examples/additional/run_cases.py
python -B examples/additional/run_cases.py --case scalar-single --json
~~~

The runner exits with zero only when all selected scenarios pass. --json includes metrics and output or a failure reason for each scenario.

Each .case.json is paired with a .expected.json containing equality checks or numeric bounds for the existing syslab.common.check helper.
Scenario notes explain the selected boundaries. Fixed seeds make inputs repeatable; expected files contain assertions rather than recorded timings.

| Scenario | Family | Size | Purpose |
| --- | --- | --- | --- |
| scalar-single | scalar | 1 | Differentiate a single-column quadratic. |
| broadcast-single | broadcast | 1 | Broadcast a one-column bias over two rows. |
| matmul-narrow | matmul | 1 | Multiply two-by-one and one-by-three matrices. |
| softmax-single | softmax | 1 | Evaluate cross-entropy with one class. |
| softmax-three | softmax | 3 | Differentiate a three-class softmax. |
| linear-odd | linear | 5 | Evaluate regression with five features. |
| mlp-odd | mlp | 7 | Differentiate a sigmoid projection with seven features. |
| broadcast-nine | broadcast | 9 | Broadcast a nine-column bias over two rows. |
| matmul-eleven | matmul | 11 | Differentiate matrix multiplication with eleven features. |
| scalar-seventeen | scalar | 17 | Differentiate a quadratic over thirty-four elements. |

Scope: NumPy-backed differentiation and graph simplification.

Supplemental inputs have their own runner, so the existing workload discovery and its 36-case suite retain their current behavior.
Bytecode generation is disabled. Reports go to standard output; storage and model artifacts use the reference code's temporary directories.

See [limitations](../../docs/LIMITATIONS.md) and [running instructions](../../docs/RUNNING.md).
