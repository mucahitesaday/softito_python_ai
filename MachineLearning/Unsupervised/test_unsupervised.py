"""Denetimsiz öğrenme projelerinin otomatik testleri."""

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


class DenetimsizOgrenmeTesti(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kumeleme = modul_yukle(KLASOR / "01_clustering_comparison" / "digital_customer_clustering.py", "customer_clustering")
        cls.rfm = modul_yukle(KLASOR / "02_rfm_segmentation" / "ecommerce_rfm.py", "ecommerce_rfm")

    def test_musteri_verisi_uretilir(self) -> None:
        veri = self.kumeleme.veriyi_uret(320)
        self.assertEqual(veri.shape, (320, 10))
        self.assertGreater(veri[self.kumeleme.OZELLIKLER].isna().sum().sum(), 0)

    def test_kume_sayisi_secimi(self) -> None:
        veri = self.kumeleme.veriyi_uret(320)
        _, X, _, _ = self.kumeleme.on_isleme(veri)
        sonuclar, en_iyi_k = self.kumeleme.k_degerlerini_incele(X)
        self.assertEqual(sonuclar["k"].tolist(), [2, 3, 4, 5, 6, 7])
        self.assertIn(en_iyi_k, range(2, 8))

    def test_dort_kumeleme_modeli(self) -> None:
        veri = self.kumeleme.veriyi_uret(320)
        _, X, _, _ = self.kumeleme.on_isleme(veri)
        sonuc = self.kumeleme.modelleri_karsilastir(X, 4, veri["known_profile"])
        self.assertEqual(set(sonuc["metrics"]["model"]), {"KMeans", "Agglomerative", "GMM", "DBSCAN"})
        self.assertGreater(sonuc["metrics"].query("model == 'KMeans'")["silhouette"].iloc[0], 0.20)

    def test_rfm_temizligi(self) -> None:
        ham = self.rfm.islemleri_uret(120)
        temiz, rapor = self.rfm.islemleri_temizle(ham)
        self.assertGreater(rapor["removed_rows"], 0)
        self.assertTrue((temiz["quantity"] > 0).all())
        self.assertFalse(temiz["is_cancelled"].any())

    def test_rfm_segmentasyonu(self) -> None:
        ham = self.rfm.islemleri_uret(180)
        temiz, _ = self.rfm.islemleri_temizle(ham)
        rfm = self.rfm.rfm_olustur(temiz)
        sonuc = self.rfm.segmentlere_ayir(rfm)
        self.assertEqual(len(rfm), 180)
        self.assertGreaterEqual(sonuc["data"]["Cluster"].nunique(), 2)
        self.assertFalse(sonuc["data"][["Recency", "Frequency", "Monetary"]].isna().any().any())


if __name__ == "__main__":
    unittest.main()
