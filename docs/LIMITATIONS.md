# Limitations

No GPU backend, custom native matmul or automatic tensor-graph fusion. The expression optimizer is separate from the dynamic Tensor graph. Backward resets all reachable gradients per invocation. Matrix multiplication is 2D only; sum supports one axis. Recursive traversal is unsuitable for extremely deep graphs.
