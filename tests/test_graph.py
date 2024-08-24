import sys, unittest, json, math, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from syslab.core import run_case
from syslab.common import validate_case,check,dumps,atomic_json,finite_numbers
from syslab.cli import evaluate,cases
from syslab.graph import simplify,evaluate,node_count
class GraphTests(unittest.TestCase):
    def test_equivalence(self):
        g=('add',('mul','x',1),('mul',2,3));h=simplify(g)
        self.assertLess(node_count(h),node_count(g))
        for x in (-5,0,8):self.assertEqual(evaluate(g,{'x':x}),evaluate(h,{'x':x}))
    def test_not_fold_variable(self):self.assertEqual(simplify(('mul','x','x')),('mul','x','x'))
