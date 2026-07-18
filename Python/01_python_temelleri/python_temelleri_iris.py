"""Python temellerini Iris veri seti üzerinde uygulayan örnek çalışma.

Bu dosya tek başına çalıştırılabilir. İlk çalıştırmada veri klasörünü ve Iris
CSV dosyasını oluşturur; ardından temel istatistikleri hesaplayıp raporlar.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from sklearn.datasets import load_iris


# Path kullanmak, Windows/macOS/Linux arasındaki dosya yolu farklarını önler.
PROJE_KLASORU = Path(__file__).resolve().parent
VERI_DOSYASI = PROJE_KLASORU / "data" / "iris.csv"
CIKTI_KLASORU = PROJE_KLASORU / "outputs"

OLCUM_SUTUNLARI = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]


def veri_setini_hazirla(dosya_yolu: Path = VERI_DOSYASI) -> Path:
    """Scikit-learn Iris verisini CSV dosyasına dönüştürür."""
    iris = load_iris()
    dosya_yolu.parent.mkdir(parents=True, exist_ok=True)

    with dosya_yolu.open("w", newline="", encoding="utf-8") as dosya:
        yazici = csv.writer(dosya)
        yazici.writerow([*OLCUM_SUTUNLARI, "species"])

        for olcumler, hedef in zip(iris.data, iris.target):
            tur_adi = str(iris.target_names[int(hedef)])
            yazici.writerow([*olcumler, tur_adi])

    return dosya_yolu


def veriyi_oku(dosya_yolu: Path = VERI_DOSYASI) -> list[dict[str, Any]]:
    """CSV dosyasını okur ve sayısal sütunları float tipine çevirir."""
    if not dosya_yolu.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {dosya_yolu}")

    kayitlar: list[dict[str, Any]] = []
    with dosya_yolu.open("r", newline="", encoding="utf-8") as dosya:
        okuyucu = csv.DictReader(dosya)
        for satir in okuyucu:
            kayit = {sutun: float(satir[sutun]) for sutun in OLCUM_SUTUNLARI}
            kayit["species"] = satir["species"]
            kayitlar.append(kayit)

    if not kayitlar:
        raise ValueError("Veri dosyasında analiz edilecek kayıt bulunamadı.")

    return kayitlar


def ture_gore_grupla(
    kayitlar: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Kayıtları çiçek türlerine göre bir sözlükte gruplar."""
    gruplar: dict[str, list[dict[str, Any]]] = {}
    for kayit in kayitlar:
        tur = str(kayit["species"])
        gruplar.setdefault(tur, []).append(kayit)
    return gruplar


class IrisAnalizi:
    """Iris kayıtları üzerinde tekrar kullanılabilir analizler yapar."""

    def __init__(self, kayitlar: list[dict[str, Any]]) -> None:
        if not kayitlar:
            raise ValueError("Analiz için en az bir kayıt gereklidir.")
        self.kayitlar = kayitlar
        self.gruplar = ture_gore_grupla(kayitlar)

    @property
    def kayit_sayisi(self) -> int:
        """Toplam kayıt sayısını döndürür."""
        return len(self.kayitlar)

    def tur_sayilari(self) -> dict[str, int]:
        """Her çiçek türünden kaç örnek bulunduğunu hesaplar."""
        sayac = Counter(str(kayit["species"]) for kayit in self.kayitlar)
        return dict(sorted(sayac.items()))

    def tur_ortalamalari(self) -> dict[str, dict[str, float]]:
        """Her tür için dört ölçümün ortalamasını hesaplar."""
        sonuc: dict[str, dict[str, float]] = {}

        for tur, tur_kayitlari in sorted(self.gruplar.items()):
            sonuc[tur] = {
                sutun: round(mean(float(kayit[sutun]) for kayit in tur_kayitlari), 3)
                for sutun in OLCUM_SUTUNLARI
            }

        return sonuc

    def en_uzun_tac_yapragi(self) -> dict[str, Any]:
        """Taç yaprağı en uzun olan kaydı bulur."""
        return max(self.kayitlar, key=lambda kayit: float(kayit["petal_length_cm"]))

    def rapor_olustur(self) -> dict[str, Any]:
        """Tüm sonuçları JSON'a uygun tek bir sözlükte toplar."""
        return {
            "veri_seti": "Iris",
            "toplam_kayit": self.kayit_sayisi,
            "tur_sayilari": self.tur_sayilari(),
            "tur_ortalamalari": self.tur_ortalamalari(),
            "en_uzun_tac_yapragi": self.en_uzun_tac_yapragi(),
        }


def sonuclari_kaydet(rapor: dict[str, Any], cikti_klasoru: Path = CIKTI_KLASORU) -> None:
    """Analiz raporunu JSON ve özet CSV dosyalarına kaydeder."""
    cikti_klasoru.mkdir(parents=True, exist_ok=True)

    json_yolu = cikti_klasoru / "analiz_raporu.json"
    with json_yolu.open("w", encoding="utf-8") as dosya:
        json.dump(rapor, dosya, ensure_ascii=False, indent=2)

    csv_yolu = cikti_klasoru / "tur_ozeti.csv"
    with csv_yolu.open("w", newline="", encoding="utf-8") as dosya:
        alanlar = ["species", "adet", *OLCUM_SUTUNLARI]
        yazici = csv.DictWriter(dosya, fieldnames=alanlar)
        yazici.writeheader()

        for tur, ortalamalar in rapor["tur_ortalamalari"].items():
            yazici.writerow(
                {
                    "species": tur,
                    "adet": rapor["tur_sayilari"][tur],
                    **ortalamalar,
                }
            )


def raporu_yazdir(rapor: dict[str, Any]) -> None:
    """Önemli analiz sonuçlarını anlaşılır biçimde terminale yazdırır."""
    print("\nIRIS VERI SETI ANALIZI")
    print("=" * 24)
    print(f"Toplam kayıt: {rapor['toplam_kayit']}")

    print("\nTür dağılımı:")
    for tur, adet in rapor["tur_sayilari"].items():
        print(f"- {tur}: {adet}")

    print("\nTürlere göre ortalama taç yaprağı uzunluğu:")
    for tur, ortalamalar in rapor["tur_ortalamalari"].items():
        print(f"- {tur}: {ortalamalar['petal_length_cm']} cm")

    en_uzun = rapor["en_uzun_tac_yapragi"]
    print(
        "\nEn uzun taç yaprağı: "
        f"{en_uzun['petal_length_cm']} cm ({en_uzun['species']})"
    )
    print(f"\nRaporlar kaydedildi: {CIKTI_KLASORU}")


def argumanlari_al() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iris veri setiyle Python temelleri")
    parser.add_argument(
        "--veriyi-yenile",
        action="store_true",
        help="Mevcut CSV dosyasını silmeden yeniden oluşturur.",
    )
    return parser.parse_args()


def main() -> None:
    args = argumanlari_al()

    try:
        if args.veriyi_yenile or not VERI_DOSYASI.exists():
            veri_setini_hazirla()
            print(f"Veri seti hazırlandı: {VERI_DOSYASI}")

        kayitlar = veriyi_oku()
        analiz = IrisAnalizi(kayitlar)
        rapor = analiz.rapor_olustur()
        sonuclari_kaydet(rapor)
        raporu_yazdir(rapor)
    except (FileNotFoundError, ValueError, OSError) as hata:
        print(f"Program çalıştırılamadı: {hata}")
        raise SystemExit(1) from hata


if __name__ == "__main__":
    main()
