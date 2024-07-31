import sys, unittest, json, math, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from syslab.core import run_case
from syslab.common import validate_case,check,dumps,atomic_json,finite_numbers
from syslab.cli import evaluate,cases
import numpy as np
from syslab.tensor import Tensor
class TensorTests(unittest.TestCase):
    def test_repeated_node(self):
        x=Tensor(3);y=x*x+x;y.backward();self.assertEqual(float(x.grad),7)
    def test_zero_power_derivative(self):
        x=Tensor([0,1]);x.power(0).sum().backward();np.testing.assert_equal(x.grad,[0,0])
    def test_repeated_slice_indices(self):
        x=Tensor([1,2]);x[[0,0]].sum().backward();np.testing.assert_equal(x.grad,[2,0])
    def test_broadcast(self):
        a=Tensor(np.ones((2,3)));b=Tensor(np.ones((1,3)));(a+b).sum().backward();np.testing.assert_equal(b.grad,[[2,2,2]])
