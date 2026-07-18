# 02 — Detaylı Python Uygulamaları

Bu bölüm Python'ın temel sözdiziminden ileri nesne yönelimli programlamaya
kadar ilerleyen beş bağımsız uygulamadan oluşur. Konular, boş ve birbirinden
kopuk örnekler yerine küçük bir e-ticaret satış veri seti üzerinden işlenir.

## Dosyalar ve konu kapsamı

| Dosya | Konular |
|---|---|
| `01_python_baslangic.py` | Değişken, veri tipi, operatör, string, koşul, döngü, liste, sözlük, fonksiyon |
| `02_koleksiyonlar_ve_fonksiyonlar.py` | CSV okuma, list/set/dict, `while`, `math`, varsayılan parametre, filtreleme |
| `03_python_alistirmalari.py` | Comprehension, `*args`, `**kwargs`, hata yönetimi, kapsamlı raporlama |
| `04_ileri_veri_yapilari.py` | Tuple, set, `enumerate`, `zip`, lambda, recursion, JSON/TXT dosya işlemleri |
| `05_nesne_yonelimli_programlama.py` | Class, kalıtım, property, classmethod, staticmethod, iterator, mixin, dunder metotlar |

## Veri seti

`data/satislar.csv` dosyasında 15 örnek e-ticaret siparişi bulunur. Sütunlar:

- Sipariş numarası ve tarih
- Şehir, kategori ve ürün
- Adet ve birim fiyat
- Müşteri puanı
- İade durumu

Bu veri, eğitim amaçlı oluşturulmuş özgün ve küçük bir veri setidir. Büyük
gerçek veri setleri EDA ve makine öğrenmesi bölümlerinde kullanılacaktır.

## Çalıştırma

Repo ana klasöründeyken dosyaları sırayla çalıştırabilirsiniz:

```bash
python Python/02_python_detayli/01_python_baslangic.py
python Python/02_python_detayli/02_koleksiyonlar_ve_fonksiyonlar.py
python Python/02_python_detayli/03_python_alistirmalari.py
python Python/02_python_detayli/04_ileri_veri_yapilari.py
python Python/02_python_detayli/05_nesne_yonelimli_programlama.py
```

Testler:

```bash
python -m unittest Python/02_python_detayli/test_python_detayli.py -v
```

`04_ileri_veri_yapilari.py` çalıştırıldığında `outputs/` klasöründe kategori
özeti ve sıralı ürün raporu oluşturulur.
