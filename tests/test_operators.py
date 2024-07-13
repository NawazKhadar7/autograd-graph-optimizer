import sys, unittest, json, math, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from syslab.core import run_case
from syslab.common import validate_case,check,dumps,atomic_json,finite_numbers
from syslab.cli import evaluate,cases
import numpy as np
from syslab.tensor import Tensor
from syslab.operators import cross_entropy
class LossTests(unittest.TestCase):
    def test_large_logits(self):
        x=Tensor([[1000,1001,-1000]]);loss=cross_entropy(x,[1]);loss.backward();self.assertTrue(np.isfinite(loss.data));self.assertAlmostEqual(float(x.grad.sum()),0)
    def test_bad_log(self):
        with self.assertRaises(ValueError):Tensor(-1).log()
