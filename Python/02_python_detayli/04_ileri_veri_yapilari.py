"""İleri veri yapıları, lambda, recursion ve dosya işlemleri."""

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

KLASOR = Path(__file__).resolve().parent
CIKTI_KLASORU = KLASOR / "outputs"


def satislari_yukle() -> list[dict]:
    yol = KLASOR / "02_koleksiyonlar_ve_fonksiyonlar.py"
    spec = spec_from_file_location("koleksiyonlar", yol)
    if spec is None or spec.loader is None:
        raise ImportError(f"Modül yüklenemedi: {yol}")
    modul = module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul.satislari_oku()


def faktoriyel(sayi: int) -> int:
    """Recursive fonksiyon örneği."""
    if sayi < 0:
        raise ValueError("Negatif sayının faktoriyeli hesaplanamaz.")
    if sayi in (0, 1):
        return 1
    return sayi * faktoriyel(sayi - 1)


def raporlari_kaydet(satislar: list[dict]) -> None:
    CIKTI_KLASORU.mkdir(parents=True, exist_ok=True)

    kategoriler = sorted({satis["kategori"] for satis in satislar})
    kategori_ozeti = {
        kategori: {
            "siparis_sayisi": sum(1 for s in satislar if s["kategori"] == kategori),
            "toplam_adet": sum(s["adet"] for s in satislar if s["kategori"] == kategori),
        }
        for kategori in kategoriler
    }

    with (CIKTI_KLASORU / "kategori_ozeti.json").open("w", encoding="utf-8") as dosya:
        json.dump(kategori_ozeti, dosya, ensure_ascii=False, indent=2)

    sirali = sorted(satislar, key=lambda satis: satis["adet"] * satis["birim_fiyat"], reverse=True)
    with (CIKTI_KLASORU / "sirali_urunler.txt").open("w", encoding="utf-8") as dosya:
        for sira, satis in enumerate(sirali, start=1):
            tutar = satis["adet"] * satis["birim_fiyat"]
            dosya.write(f"{sira}. {satis['urun']} — {tutar:.2f} TL\n")


def main() -> None:
    satislar = satislari_yukle()

    # Tuple ve unpacking
    siparis_urun_ciftleri = [(s["siparis_id"], s["urun"]) for s in satislar]
    ilk_id, ilk_urun = siparis_urun_ciftleri[0]

    # Set işlemleri
    istanbul_urunleri = {s["urun"] for s in satislar if s["sehir"] == "Istanbul"}
    ankara_urunleri = {s["urun"] for s in satislar if s["sehir"] == "Ankara"}

    # zip: iki listeyi eşleştirme
    kategori_adlari = sorted({s["kategori"] for s in satislar})
    kategori_kodlari = ["ELK", "EV", "KTP", "SPR"]
    kategori_kod_sozlugu = dict(zip(kategori_adlari, kategori_kodlari))

    raporlari_kaydet(satislar)

    print("İLERİ VERİ YAPILARI")
    print("=" * 21)
    print(f"İlk tuple: ({ilk_id}, {ilk_urun})")
    print(f"İstanbul ürünleri: {sorted(istanbul_urunleri)}")
    print(f"İstanbul ve Ankara ortak ürün sayısı: {len(istanbul_urunleri & ankara_urunleri)}")
    print(f"Kategori kodları: {kategori_kod_sozlugu}")
    print(f"5! = {faktoriyel(5)}")
    print(f"Raporlar kaydedildi: {CIKTI_KLASORU}")


if __name__ == "__main__":
    main()
