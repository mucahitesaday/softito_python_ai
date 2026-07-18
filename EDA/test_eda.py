"""EDA bölümünün veri ve özellik üretim testleri."""

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

KLASOR = Path(__file__).resolve().parent


def modul_yukle(goreli_yol: str, ad: str):
    spec = importlib.util.spec_from_file_location(ad, KLASOR / goreli_yol)
    if spec is None or spec.loader is None:
        raise ImportError(goreli_yol)
    modul = importlib.util.module_from_spec(spec)
    sys.modules[ad] = modul
    spec.loader.exec_module(modul)
    return modul


class EDATesti(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.yukleme = modul_yukle("01_veri_yukleme_genel_bakis/veri_yukleme.py", "eda_yukleme")
        cls.temizleme = modul_yukle("02_veri_temizleme/veri_temizleme.py", "eda_temizleme")
        cls.tek = modul_yukle("03_tek_degiskenli_analiz/tek_degiskenli_analiz.py", "eda_tek")
        cls.cift = modul_yukle("04_cift_degiskenli_analiz/cift_degiskenli_analiz.py", "eda_cift")
        cls.feature = modul_yukle("05_feature_engineering/feature_engineering.py", "eda_feature")

    def test_veri_boyutu(self) -> None:
        df = self.yukleme.veriyi_yukle()
        self.assertEqual(df.shape, (344, 8))

    def test_eksik_degerler_temizlenir(self) -> None:
        df = self.yukleme.veriyi_yukle()
        temiz = self.temizleme.veriyi_temizle(df)
        self.assertEqual(int(temiz.isna().sum().sum()), 0)

    def test_tek_ve_cift_analiz_verisi(self) -> None:
        self.assertEqual(self.tek.veriyi_hazirla().shape[0], 344)
        self.assertEqual(self.cift.veriyi_hazirla().shape[0], 344)

    def test_yeni_ozellikler_uretilir(self) -> None:
        df = pd.read_csv(KLASOR / "data" / "penguins.csv", na_values=["NA", ""])
        sonuc = self.feature.ozellik_uret(df)
        beklenen = {"body_mass_kg", "bill_area_index", "flipper_mass_ratio", "body_size_category", "sex_encoded"}
        self.assertTrue(beklenen.issubset(sonuc.columns))

    def test_kategorik_kodlama(self) -> None:
        df = pd.read_csv(KLASOR / "data" / "penguins.csv", na_values=["NA", ""])
        sonuc = self.feature.ozellik_uret(df)
        self.assertTrue({"island_Biscoe", "island_Dream", "island_Torgersen"}.issubset(sonuc.columns))


if __name__ == "__main__":
    unittest.main()
