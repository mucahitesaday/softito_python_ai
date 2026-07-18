"""Logistic Regression projelerinin otomatik testleri."""

import importlib.util
import sys
import unittest
from pathlib import Path

KLASOR = Path(__file__).resolve().parent


def modul_yukle(yol: Path, ad: str):
    spec = importlib.util.spec_from_file_location(ad, yol)
    if spec is None or spec.loader is None:
        raise ImportError(yol)
    modul = importlib.util.module_from_spec(spec)
    sys.modules[ad] = modul
    spec.loader.exec_module(modul)
    return modul


class LogisticRegresyonTesti(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kanser = modul_yukle(
            KLASOR / "breast_cancer_diagnosis" / "breast_cancer_logistic.py",
            "breast_cancer_logistic",
        )
        cls.sarap = modul_yukle(
            KLASOR / "wine_classification" / "wine_logistic.py",
            "wine_logistic",
        )

    def test_kanser_veri_boyutu(self) -> None:
        X, y = self.kanser.veriyi_yukle()
        self.assertEqual(X.shape, (569, 30))
        self.assertEqual(set(y.unique()), {0, 1})

    def test_kanser_model_performansi(self) -> None:
        X, y = self.kanser.veriyi_yukle()
        sonuc = self.kanser.modeli_egit(X, y)
        self.assertGreater(sonuc["metrikler"]["accuracy"], 0.90)
        self.assertGreater(sonuc["metrikler"]["recall"], 0.85)

    def test_esik_analizi(self) -> None:
        X, y = self.kanser.veriyi_yukle()
        sonuc = self.kanser.modeli_egit(X, y)
        self.assertEqual(sonuc["esik_sonuclari"]["threshold"].tolist(), [0.3, 0.5, 0.7])

    def test_sarap_veri_boyutu(self) -> None:
        X, y, siniflar = self.sarap.veriyi_yukle()
        self.assertEqual(X.shape, (178, 13))
        self.assertEqual(len(siniflar), 3)
        self.assertEqual(y.nunique(), 3)

    def test_sarap_model_performansi(self) -> None:
        X, y, siniflar = self.sarap.veriyi_yukle()
        sonuc = self.sarap.modeli_egit(X, y, siniflar)
        self.assertGreater(sonuc["metrikler"]["accuracy"], 0.90)
        self.assertEqual(sonuc["katsayilar"].shape, (13, 3))
        self.assertEqual(
            set(sonuc["metrikler"]),
            {"accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro", "log_loss"},
        )


if __name__ == "__main__":
    unittest.main()
