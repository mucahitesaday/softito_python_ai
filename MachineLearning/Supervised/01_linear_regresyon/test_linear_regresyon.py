"""Linear Regression projelerinin otomatik testleri."""

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


class LinearRegresyonTesti(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tip = modul_yukle(
            KLASOR / "restaurant_tip_prediction" / "tip_regression.py", "tip_regression"
        )
        cls.diabetes = modul_yukle(
            KLASOR / "diabetes_progression" / "diabetes_regression.py", "diabetes_regression"
        )

    def test_tips_veri_boyutu(self) -> None:
        df = self.tip.veriyi_yukle()
        self.assertEqual(df.shape, (244, 7))
        self.assertEqual(int(df.isna().sum().sum()), 0)

    def test_tips_modelleri_tahmin_uretir(self) -> None:
        sonuc = self.tip.modelleri_egit(self.tip.veriyi_yukle())
        self.assertEqual(len(sonuc["test"]), len(sonuc["coklu_tahmin"]))
        self.assertGreater(sonuc["basit_metrikler"]["r2"], 0.25)

    def test_diabetes_veri_boyutu(self) -> None:
        X, y = self.diabetes.veriyi_yukle()
        self.assertEqual(X.shape, (442, 10))
        self.assertEqual(len(y), 442)

    def test_diabetes_coklu_model(self) -> None:
        X, y = self.diabetes.veriyi_yukle()
        sonuc = self.diabetes.modelleri_egit(X, y)
        self.assertGreater(sonuc["coklu_metrikler"]["r2"], 0.30)
        self.assertEqual(len(sonuc["katsayilar"]), 10)

    def test_metrikler_beklenen_alanlari_icerir(self) -> None:
        X, y = self.diabetes.veriyi_yukle()
        sonuc = self.diabetes.modelleri_egit(X, y)
        self.assertEqual(set(sonuc["coklu_metrikler"]), {"r2", "mae", "rmse"})


if __name__ == "__main__":
    unittest.main()
