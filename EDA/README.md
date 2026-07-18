# Palmer Penguins — Keşifsel Veri Analizi

Bu bölümde Antarktika'daki üç penguen türüne ait ölçümler kullanılarak uçtan
uca keşifsel veri analizi (EDA) yapılır. Veri setinde 344 penguen ve 8 sütun
bulunur.

## Veri seti

Palmer Penguins veri seti Dr. Kristen Gorman ve Palmer Station LTER programı
tarafından toplanmıştır. Veri CC0 lisansıyla yayımlanmıştır.

- Resmî proje: https://allisonhorst.github.io/palmerpenguins/
- Kaggle alternatifi: https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data
- Yerel dosya: `data/penguins.csv`

Veri repoya eklendiği için Kaggle hesabına veya internet bağlantısına ihtiyaç
yoktur.

## Bölümler

| No | Konu | Üretilen sonuç |
|---|---|---|
| 01 | Veri yükleme ve genel bakış | Eksik değer grafikleri |
| 02 | Veri temizleme | Temiz CSV ve aykırı değer boxplotları |
| 03 | Tek değişkenli analiz | Sayısal ve kategorik dağılımlar |
| 04 | Çift değişkenli analiz | Korelasyon, scatter, çapraz tablo ve boxplot |
| 05 | Feature engineering | Yeni özellikler ve işlenmiş CSV |

Her klasör kendi Python dosyasına, açıklamasına ve `figures/` klasörüne
sahiptir. Dosyalar repo ana klasöründen sırayla çalıştırılabilir:

```bash
python EDA/01_veri_yukleme_genel_bakis/veri_yukleme.py
python EDA/02_veri_temizleme/veri_temizleme.py
python EDA/03_tek_degiskenli_analiz/tek_degiskenli_analiz.py
python EDA/04_cift_degiskenli_analiz/cift_degiskenli_analiz.py
python EDA/05_feature_engineering/feature_engineering.py
```

Test:

```bash
python -m unittest EDA/test_eda.py -v
```
