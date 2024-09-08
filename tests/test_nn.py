import sys, unittest, json, math, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from syslab.core import run_case
from syslab.common import validate_case,check,dumps,atomic_json,finite_numbers
from syslab.cli import evaluate,cases
import numpy as np
from syslab.tensor import Tensor
from syslab.nn import MLP
class NetworkTests(unittest.TestCase):
    def test_network_shapes(self):
        m=MLP(3,4,2);y=m(Tensor(np.ones((5,3))));y.sum().backward();self.assertEqual(y.data.shape,(5,2));self.assertEqual(len(m.parameters()),4)
    def test_no_vector_matmul(self):
        with self.assertRaises(ValueError):Tensor([1,2])@Tensor([2,3])
