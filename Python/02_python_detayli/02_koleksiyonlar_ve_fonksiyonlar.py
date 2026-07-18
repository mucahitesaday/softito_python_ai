"""Koleksiyonları, döngüleri ve fonksiyonları satış CSV'siyle uygular."""

from __future__ import annotations

import csv
import math
from pathlib import Path

VERI_DOSYASI = Path(__file__).resolve().parent / "data" / "satislar.csv"


def satislari_oku(dosya_yolu: Path = VERI_DOSYASI) -> list[dict]:
    """CSV satırlarını uygun Python veri tiplerine dönüştürür."""
    satislar = []
    with dosya_yolu.open(encoding="utf-8", newline="") as dosya:
        for satir in csv.DictReader(dosya):
            satir["adet"] = int(satir["adet"])
            satir["birim_fiyat"] = float(satir["birim_fiyat"])
            satir["puan"] = int(satir["puan"])
            satir["iade"] = satir["iade"] == "evet"
            satislar.append(satir)
    return satislar


def siparis_tutari(satis: dict) -> float:
    return round(satis["adet"] * satis["birim_fiyat"], 2)


def kategori_ozeti(satislar: list[dict]) -> dict[str, float]:
    """Sözlük kullanarak kategori bazında ciro toplar; iadeleri hariç tutar."""
    ozet: dict[str, float] = {}
    for satis in satislar:
        if satis["iade"]:
            continue
        kategori = satis["kategori"]
        ozet[kategori] = ozet.get(kategori, 0.0) + siparis_tutari(satis)
    return {kategori: round(ciro, 2) for kategori, ciro in sorted(ozet.items())}


def hedefe_kac_siparis_kaldi(mevcut_ciro: float, hedef: float, ortalama: float) -> int:
    """math.ceil ile hedefe ulaşmak için gereken tahmini sipariş sayısını bulur."""
    kalan = max(0.0, hedef - mevcut_ciro)
    return math.ceil(kalan / ortalama) if ortalama > 0 else 0


def ilk_yuksek_puanli_siparis(satislar: list[dict], baslangic: int = 0) -> dict | None:
    """while döngüsüyle 5 puanlı ilk siparişi arar."""
    indeks = baslangic
    while indeks < len(satislar):
        if satislar[indeks]["puan"] == 5:
            return satislar[indeks]
        indeks += 1
    return None


def main() -> None:
    satislar = satislari_oku()
    sehirler = {satis["sehir"] for satis in satislar}
    tutarlar = [siparis_tutari(satis) for satis in satislar if not satis["iade"]]
    ozet = kategori_ozeti(satislar)

    print("KOLEKSİYONLAR VE FONKSİYONLAR")
    print("=" * 33)
    print(f"Satış sayısı: {len(satislar)}")
    print(f"Farklı şehirler: {', '.join(sorted(sehirler))}")
    print(f"İade hariç toplam ciro: {sum(tutarlar):.2f} TL")
    print(f"Ortalama sipariş: {sum(tutarlar) / len(tutarlar):.2f} TL")

    print("\nKategori ciroları:")
    for kategori, ciro in ozet.items():
        print(f"- {kategori}: {ciro:.2f} TL")

    yuksek_puan = ilk_yuksek_puanli_siparis(satislar)
    print(f"\nİlk 5 puanlı sipariş: {yuksek_puan['siparis_id']}")
    print(
        "20.000 TL hedefi için tahmini ek sipariş: "
        f"{hedefe_kac_siparis_kaldi(sum(tutarlar), 20000, sum(tutarlar) / len(tutarlar))}"
    )


if __name__ == "__main__":
    main()
