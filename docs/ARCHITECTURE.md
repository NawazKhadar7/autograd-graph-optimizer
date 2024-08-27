# Architecture

Tensor operations → dynamic DAG → reverse traversal → parameter gradients. Pure expression AST → constant folding → equivalence check. NumPy executes forward array operations.
