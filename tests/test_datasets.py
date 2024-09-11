import sys, unittest, json, math, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from syslab.core import run_case
from syslab.common import validate_case,check,dumps,atomic_json,finite_numbers
from syslab.cli import evaluate,cases
import numpy as np
from syslab.datasets import regression
class DataTests(unittest.TestCase):
    def test_seed(self):
        a,b=regression(4,3,2);c,d=regression(4,3,2);np.testing.assert_equal(a,c);np.testing.assert_equal(b,d)
    def test_shapes(self):
        a,b=regression(4,3,2);self.assertEqual(a.shape,(4,3));self.assertEqual(b.shape,(4,1))
