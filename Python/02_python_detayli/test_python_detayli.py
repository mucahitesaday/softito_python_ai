"""Detaylı Python bölümünün temel davranış testleri."""

import importlib.util
import sys
import unittest
from pathlib import Path

KLASOR = Path(__file__).resolve().parent


def modul_yukle(dosya_adi: str, modul_adi: str):
    spec = importlib.util.spec_from_file_location(modul_adi, KLASOR / dosya_adi)
    if spec is None or spec.loader is None:
        raise ImportError(dosya_adi)
    modul = importlib.util.module_from_spec(spec)
    # dataclass gibi araçlar çalışırken modülün sys.modules içinde olması gerekir.
    sys.modules[modul_adi] = modul
    spec.loader.exec_module(modul)
    return modul


class DetayliPythonTesti(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baslangic = modul_yukle("01_python_baslangic.py", "baslangic")
        cls.koleksiyonlar = modul_yukle("02_koleksiyonlar_ve_fonksiyonlar.py", "koleksiyonlar_test")
        cls.alistirmalar = modul_yukle("03_python_alistirmalari.py", "alistirmalar")
        cls.ileri = modul_yukle("04_ileri_veri_yapilari.py", "ileri")
        cls.oop = modul_yukle("05_nesne_yonelimli_programlama.py", "oop")

    def test_baslangic_hesaplari(self) -> None:
        self.assertEqual(self.baslangic.sepet_tutari(100, 2, 0.10), 180.0)
        self.assertEqual(self.baslangic.stok_mesaji(0), "Tükendi")

    def test_csv_verisi_okunur(self) -> None:
        satislar = self.koleksiyonlar.satislari_oku()
        self.assertEqual(len(satislar), 15)
        self.assertIsInstance(satislar[0]["birim_fiyat"], float)

    def test_hata_yonetimi(self) -> None:
        self.assertEqual(self.alistirmalar.guvenli_indirimli_fiyat(1000, "0.2"), 800.0)
        with self.assertRaises(ValueError):
            self.alistirmalar.guvenli_indirimli_fiyat(1000, "yanlış")

    def test_recursive_faktoriyel(self) -> None:
        self.assertEqual(self.ileri.faktoriyel(5), 120)
        with self.assertRaises(ValueError):
            self.ileri.faktoriyel(-1)

    def test_oop_siparis(self) -> None:
        urun = self.oop.Urun("Test Ürünü", "Test", 100)
        siparis = self.oop.Siparis("T001", urun, 3)
        self.assertEqual(siparis.toplam_tutar, 300)
        siparis.adet = 2
        self.assertEqual(siparis.toplam_tutar, 200)
        with self.assertRaises(ValueError):
            siparis.adet = 0


if __name__ == "__main__":
    unittest.main()
