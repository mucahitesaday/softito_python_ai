"""Python temelleri Iris çalışmasının basit doğrulama testleri."""

import tempfile
import unittest
from pathlib import Path

from python_temelleri_iris import IrisAnalizi, sonuclari_kaydet


class IrisAnaliziTesti(unittest.TestCase):
    def setUp(self) -> None:
        self.kayitlar = [
            {
                "sepal_length_cm": 5.0,
                "sepal_width_cm": 3.0,
                "petal_length_cm": 1.0,
                "petal_width_cm": 0.2,
                "species": "setosa",
            },
            {
                "sepal_length_cm": 6.0,
                "sepal_width_cm": 3.0,
                "petal_length_cm": 5.0,
                "petal_width_cm": 1.8,
                "species": "virginica",
            },
        ]

    def test_kayit_ve_tur_sayilari(self) -> None:
        analiz = IrisAnalizi(self.kayitlar)
        self.assertEqual(analiz.kayit_sayisi, 2)
        self.assertEqual(analiz.tur_sayilari(), {"setosa": 1, "virginica": 1})

    def test_en_uzun_tac_yapragi(self) -> None:
        analiz = IrisAnalizi(self.kayitlar)
        self.assertEqual(analiz.en_uzun_tac_yapragi()["species"], "virginica")

    def test_rapor_dosyalari_olusturulur(self) -> None:
        analiz = IrisAnalizi(self.kayitlar)
        with tempfile.TemporaryDirectory() as gecici_klasor:
            cikti = Path(gecici_klasor)
            sonuclari_kaydet(analiz.rapor_olustur(), cikti)
            self.assertTrue((cikti / "analiz_raporu.json").exists())
            self.assertTrue((cikti / "tur_ozeti.csv").exists())


if __name__ == "__main__":
    unittest.main()
