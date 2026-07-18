"""Anomali tespiti uygulamalarının otomatik testleri."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

KLASOR = Path(__file__).resolve().parent


def modul_yukle(yol: Path, ad: str):
    spec = importlib.util.spec_from_file_location(ad, yol)
    if spec is None or spec.loader is None:
        raise ImportError(yol)
    modul = importlib.util.module_from_spec(spec)
    sys.modules[ad] = modul
    spec.loader.exec_module(modul)
    return modul


class AnomaliTespitiTesti(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.isolation = modul_yukle(KLASOR / "01_isolation_forest" / "production_sensor_isolation.py", "production_isolation")
        cls.ocsvm = modul_yukle(KLASOR / "02_one_class_svm" / "network_traffic_ocsvm.py", "network_ocsvm")
        cls.copod = modul_yukle(KLASOR / "03_copod" / "transaction_copod.py", "transaction_copod")

    def test_isolation_veri_yapisi(self) -> None:
        egitim, test = self.isolation.veriyi_uret(300, 120, 30)
        self.assertEqual(len(egitim), 300)
        self.assertEqual(test["is_anomaly"].sum(), 30)
        self.assertFalse(egitim[self.isolation.OZELLIKLER].isna().any().any())

    def test_isolation_model_basarisi(self) -> None:
        egitim, test = self.isolation.veriyi_uret(500, 180, 40)
        sonuc = self.isolation.modeli_egit(egitim, test)
        self.assertGreater(sonuc["metrics"]["roc_auc"], 0.90)
        self.assertGreater(sonuc["metrics"]["recall"], 0.70)

    def test_ocsvm_veri_bolumleri(self) -> None:
        egitim, dogrulama, test = self.ocsvm.veri_setlerini_uret(300, 80, 20, 120, 30)
        self.assertEqual((len(egitim), len(dogrulama), len(test)), (300, 100, 150))
        self.assertEqual(test["is_anomaly"].sum(), 30)

    def test_ocsvm_model_basarisi(self) -> None:
        egitim, dogrulama, test = self.ocsvm.veri_setlerini_uret(500, 120, 30, 180, 45)
        sonuc = self.ocsvm.modeli_egit(egitim, dogrulama, test)
        self.assertGreater(sonuc["metrics"]["roc_auc"], 0.90)
        self.assertIn(sonuc["best_parameters"]["nu"], (0.01, 0.03, 0.05, 0.08, 0.12))

    def test_copod_katki_matrisi(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 4))
        model = self.copod.EmpiricalCOPOD(0.05).fit(X)
        katkilar = model.feature_contributions(X[:12])
        self.assertEqual(katkilar.shape, (12, 4))
        self.assertTrue((katkilar >= 0).all())

    def test_uc_anomali_modeli(self) -> None:
        egitim, test = self.copod.veriyi_uret(600, 220, 50)
        sonuc = self.copod.modelleri_karsilastir(egitim, test)
        self.assertEqual(set(sonuc["metrics"]["model"]), {"COPOD", "IsolationForest", "OneClassSVM"})
        self.assertGreater(sonuc["metrics"]["roc_auc"].min(), 0.80)


if __name__ == "__main__":
    unittest.main()
