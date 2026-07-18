"""Klasik makine öğrenmesi paketinin testleri."""
import importlib.util,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module; spec.loader.exec_module(module); return module
class ClassicMLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.poly=load(ROOT/"01_polynomial_regression"/"energy_polynomial.py","energy_polynomial")
        cls.tree=load(ROOT/"02_decision_tree"/"promotion_tree.py","promotion_tree")
        cls.knn=load(ROOT/"03_knn_naive_bayes"/"digits_comparison.py","digits_comparison")
        cls.svm=load(ROOT/"04_svm"/"moons_svm.py","moons_svm")
        cls.boost=load(ROOT/"05_boosting"/"maintenance_boosting.py","maintenance_boosting")
    def test_polynomial(self):
        df=self.poly.veriyi_uret(300); s=self.poly.modelleri_karsilastir(df); self.assertIn(s["best_degree"],(2,3,5)); self.assertGreater(s["metrics"].query("split=='test'").r2.max(),.9)
    def test_decision_tree(self):
        df=self.tree.veriyi_uret(700); s=self.tree.modeli_egit(df); self.assertGreater(s["metrics"]["roc_auc"],.75); self.assertEqual(len(s["importance"]),11)
    def test_digits(self):
        X,y,_=self.knn.veriyi_yukle(); s=self.knn.modeli_egit(X,y); self.assertGreater(s["comparison"].accuracy.max(),.9); self.assertIn(s["best_k"],(1,3,5,7,9))
    def test_svm(self):
        X,y=self.svm.veriyi_uret(500); s=self.svm.modelleri_egit(X,y); self.assertGreater(s["metrics"].accuracy.max(),.85); self.assertEqual(len(s["metrics"]),4)
    def test_boosting(self):
        X,y=self.boost.veriyi_uret(900); s=self.boost.modelleri_egit(X,y); self.assertGreater(s["metrics"].roc_auc.max(),.85); self.assertIn("AdaBoost",s["models"])
if __name__=="__main__": unittest.main()
