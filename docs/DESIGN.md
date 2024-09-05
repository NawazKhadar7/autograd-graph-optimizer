# Design

Restrict operator semantics and explicitly reject nonfinite tensors. Stable cross entropy subtracts a row maximum treated as a constant; softmax shift invariance makes the derivative correct.
