"""Comprehension, esnek parametreler ve hata yönetimi uygulamaları."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

KLASOR = Path(__file__).resolve().parent


def veri_modulunu_yukle():
    """Rakamla başlayan ders dosyasını güvenli biçimde modül olarak yükler."""
    yol = KLASOR / "02_koleksiyonlar_ve_fonksiyonlar.py"
    spec = spec_from_file_location("koleksiyonlar", yol)
    if spec is None or spec.loader is None:
        raise ImportError(f"Modül yüklenemedi: {yol}")
    modul = module_from_spec(spec)
    spec.loader.exec_module(modul)
    return modul


def ortalama_puan(*puanlar: int) -> float:
    """*args ile değişken sayıda puanın ortalamasını hesaplar."""
    if not puanlar:
        raise ValueError("En az bir puan gereklidir.")
    if any(puan < 1 or puan > 5 for puan in puanlar):
        raise ValueError("Puanlar 1 ile 5 arasında olmalıdır.")
    return round(sum(puanlar) / len(puanlar), 2)


def filtre_olustur(**kriterler) -> str:
    """**kwargs ile verilen filtreleri okunabilir metne dönüştürür."""
    return ", ".join(f"{anahtar}={deger}" for anahtar, deger in kriterler.items())


def guvenli_indirimli_fiyat(fiyat: float, oran_metni: str) -> float:
    """Hatalı kullanıcı girdilerini try/except ile yakalar."""
    try:
        oran = float(oran_metni)
        if not 0 <= oran <= 1:
            raise ValueError("İndirim oranı 0 ile 1 arasında olmalıdır.")
        return round(fiyat * (1 - oran), 2)
    except TypeError as hata:
        raise ValueError("İndirim oranı metin veya sayı olmalıdır.") from hata


def main() -> None:
    veri_modulu = veri_modulunu_yukle()
    satislar = veri_modulu.satislari_oku()

    # List comprehension: iade edilmeyen yüksek puanlı siparişler
    memnun_siparisler = [
        satis["siparis_id"]
        for satis in satislar
        if satis["puan"] >= 4 and not satis["iade"]
    ]

    # Dict comprehension: şehirlerin sipariş sayıları
    sehirler = sorted({satis["sehir"] for satis in satislar})
    sehir_sayilari = {
        sehir: sum(1 for satis in satislar if satis["sehir"] == sehir)
        for sehir in sehirler
    }

    print("PYTHON ALIŞTIRMALARI")
    print("=" * 21)
    print(f"Memnun müşteri siparişleri: {memnun_siparisler}")
    print(f"Şehir sipariş sayıları: {sehir_sayilari}")
    print(f"Puan ortalaması: {ortalama_puan(*(s['puan'] for s in satislar))}")
    print(f"Aktif filtre: {filtre_olustur(sehir='Istanbul', puan=5, iade=False)}")

    try:
        print(f"%15 indirimli fiyat: {guvenli_indirimli_fiyat(1000, '0.15')} TL")
        guvenli_indirimli_fiyat(1000, "hatalı")
    except ValueError as hata:
        print(f"Kontrollü hata mesajı: {hata}")


if __name__ == "__main__":
    main()
