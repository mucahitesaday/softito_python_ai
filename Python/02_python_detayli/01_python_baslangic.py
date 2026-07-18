"""Python başlangıç konularını küçük bir ürün veri setiyle gösterir."""

URUNLER = [
    {"ad": "Kablosuz Kulaklık", "kategori": "Elektronik", "fiyat": 1250.0, "stok": 18},
    {"ad": "Masa Lambası", "kategori": "Ev", "fiyat": 640.0, "stok": 7},
    {"ad": "Python ile Veri Bilimi", "kategori": "Kitap", "fiyat": 310.0, "stok": 25},
    {"ad": "Yoga Matı", "kategori": "Spor", "fiyat": 475.0, "stok": 0},
]


def sepet_tutari(fiyat: float, adet: int, indirim_orani: float = 0.0) -> float:
    """Aritmetik operatörlerle indirimli sepet tutarını hesaplar."""
    ara_toplam = fiyat * adet
    indirim = ara_toplam * indirim_orani
    return round(ara_toplam - indirim, 2)


def stok_mesaji(stok: int) -> str:
    """Koşullu ifadelerle stok seviyesini sınıflandırır."""
    if stok == 0:
        return "Tükendi"
    if stok < 10:
        return "Kritik stok"
    return "Stok yeterli"


def kategoriye_gore_filtrele(kategori: str) -> list[dict]:
    """Döngü ve liste kullanarak istenen kategorideki ürünleri döndürür."""
    bulunanlar = []
    for urun in URUNLER:
        if urun["kategori"].lower() == kategori.lower():
            bulunanlar.append(urun)
    return bulunanlar


def main() -> None:
    magaza_adi = "VeriSepeti"
    aktif = True
    urun_sayisi = len(URUNLER)
    ortalama_fiyat = sum(urun["fiyat"] for urun in URUNLER) / urun_sayisi

    print("PYTHON BAŞLANGIÇ — ÜRÜN VERİSİ")
    print("=" * 34)
    print(f"Mağaza: {magaza_adi.upper()}")
    print(f"Aktif mi? {aktif} ({type(aktif).__name__})")
    print(f"Ürün sayısı: {urun_sayisi} ({type(urun_sayisi).__name__})")
    print(f"Ortalama fiyat: {ortalama_fiyat:.2f} TL")

    print("\nÜrünler:")
    for sira, urun in enumerate(URUNLER, start=1):
        print(
            f"{sira}. {urun['ad']} | {urun['kategori']} | "
            f"{urun['fiyat']:.2f} TL | {stok_mesaji(urun['stok'])}"
        )

    kitaplar = kategoriye_gore_filtrele("Kitap")
    print(f"\nKitap kategorisinde {len(kitaplar)} ürün var.")
    print(f"2 adet kulaklık (%10 indirimli): {sepet_tutari(1250, 2, 0.10)} TL")


if __name__ == "__main__":
    main()
