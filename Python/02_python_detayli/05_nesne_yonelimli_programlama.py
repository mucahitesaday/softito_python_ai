"""E-ticaret verisiyle ileri nesne yönelimli programlama uygulaması."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


class SozlugeDonusturMixin:
    """Nesnenin herkese açık alanlarını sözlüğe dönüştürür."""

    def sozluge_donustur(self) -> dict:
        return {anahtar: deger for anahtar, deger in vars(self).items() if not anahtar.startswith("_")}


@dataclass
class Urun:
    ad: str
    kategori: str
    fiyat: float

    def __post_init__(self) -> None:
        if self.fiyat <= 0:
            raise ValueError("Ürün fiyatı pozitif olmalıdır.")

    def __str__(self) -> str:
        return f"{self.ad} ({self.fiyat:.2f} TL)"

    def __lt__(self, diger: Urun) -> bool:
        return self.fiyat < diger.fiyat


class Siparis(SozlugeDonusturMixin):
    toplam_siparis = 0

    def __init__(self, siparis_id: str, urun: Urun, adet: int, tarih: date | None = None):
        if not self.gecerli_adet(adet):
            raise ValueError("Adet pozitif bir tam sayı olmalıdır.")
        self.siparis_id = siparis_id
        self.urun = urun
        self._adet = adet
        self.tarih = tarih or date.today()
        Siparis.toplam_siparis += 1

    @property
    def adet(self) -> int:
        return self._adet

    @adet.setter
    def adet(self, yeni_adet: int) -> None:
        if not self.gecerli_adet(yeni_adet):
            raise ValueError("Adet pozitif bir tam sayı olmalıdır.")
        self._adet = yeni_adet

    @property
    def toplam_tutar(self) -> float:
        return round(self.urun.fiyat * self.adet, 2)

    @classmethod
    def hizli_olustur(cls, siparis_id: str, urun_adi: str, fiyat: float) -> Siparis:
        return cls(siparis_id, Urun(urun_adi, "Belirtilmedi", fiyat), 1)

    @staticmethod
    def gecerli_adet(adet: int) -> bool:
        return isinstance(adet, int) and not isinstance(adet, bool) and adet > 0

    def __str__(self) -> str:
        return f"[{self.siparis_id}] {self.urun.ad} x {self.adet} = {self.toplam_tutar:.2f} TL"

    def __repr__(self) -> str:
        return f"Siparis({self.siparis_id!r}, {self.urun!r}, {self.adet!r})"

    def __eq__(self, diger: object) -> bool:
        if not isinstance(diger, Siparis):
            return NotImplemented
        return self.siparis_id == diger.siparis_id


class IndirimliSiparis(Siparis):
    """Kalıtım ve metot ezme örneği."""

    def __init__(self, *args, indirim_orani: float, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= indirim_orani <= 1:
            raise ValueError("İndirim oranı 0 ile 1 arasında olmalıdır.")
        self.indirim_orani = indirim_orani

    @property
    def toplam_tutar(self) -> float:
        normal_tutar = super().toplam_tutar
        return round(normal_tutar * (1 - self.indirim_orani), 2)


class SiparisKoleksiyonu:
    """Iterator, len ve contains dunder metotlarını gösterir."""

    def __init__(self, siparisler: list[Siparis] | None = None):
        self.siparisler = siparisler or []

    def ekle(self, siparis: Siparis) -> None:
        if siparis in self.siparisler:
            raise ValueError("Aynı sipariş numarası tekrar eklenemez.")
        self.siparisler.append(siparis)

    def __iter__(self):
        return iter(self.siparisler)

    def __len__(self) -> int:
        return len(self.siparisler)

    def __contains__(self, siparis_id: str) -> bool:
        return any(siparis.siparis_id == siparis_id for siparis in self.siparisler)


def main() -> None:
    kulaklik = Urun("Kablosuz Kulaklık", "Elektronik", 1250)
    kitap = Urun("Python ile Veri Bilimi", "Kitap", 310)

    s1 = Siparis("S101", kulaklik, 2)
    s2 = IndirimliSiparis("S102", kitap, 3, indirim_orani=0.10)
    s3 = Siparis.hizli_olustur("S103", "Masa Lambası", 640)

    koleksiyon = SiparisKoleksiyonu([s1, s2])
    koleksiyon.ekle(s3)

    print("NESNE YÖNELİMLİ PROGRAMLAMA")
    print("=" * 29)
    for siparis in koleksiyon:
        print(siparis)

    print(f"\nToplam nesne sayısı: {Siparis.toplam_siparis}")
    print(f"Koleksiyondaki sipariş sayısı: {len(koleksiyon)}")
    print(f"S102 koleksiyonda mı? {'S102' in koleksiyon}")
    print(f"En pahalı ürün: {max([kulaklik, kitap])}")
    print(f"Mixin çıktısı: {s1.sozluge_donustur()}")


if __name__ == "__main__":
    main()
