import sys, unittest, json, math, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from syslab.core import run_case
from syslab.common import validate_case,check,dumps,atomic_json,finite_numbers
from syslab.cli import evaluate,cases
from syslab.tensor import Tensor
from syslab.optim import SGD
class OptimizerTests(unittest.TestCase):
    def test_descent(self):
        x=Tensor(3);initial=float((x*x).data);opt=SGD([x],.1)
        for _ in range(5):(x*x).backward();opt.step()
        self.assertLess(float((x*x).data),initial)
    def test_bad_lr(self):
        with self.assertRaises(ValueError):SGD([],0)
