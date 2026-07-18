# Restoran Bahşiş Tahmini

## Amaç

Bir restoran hesabında bırakılacak bahşiş miktarını tahmin etmek ve tek bir
değişken kullanan model ile bütün sipariş özelliklerini kullanan modeli
karşılaştırmaktır.

1. Basit regresyon: yalnızca `total_bill`
2. Çoklu regresyon: hesap, kişi sayısı, cinsiyet, sigara durumu, gün ve öğün

## Veri seti

Seaborn `tips` veri setinde 244 restoran hesabı bulunur. Hedef sütun `tip`tir.
Veri dosyası projeye eklenmiştir; internet veya API anahtarı gerekmez.

Kaynak: https://github.com/mwaskom/seaborn-data/blob/master/tips.csv

## Çalıştırma

```bash
python MachineLearning/Supervised/01_linear_regresyon/restaurant_tip_prediction/tip_regression.py
```

Üretilenler:

- Model metrikleri (`results/metrics.json`)
- Katsayı tablosu (`results/coefficients.csv`)
- Korelasyon, regresyon, katsayı, gerçek–tahmin ve artık grafikleri

R², modelin hedefteki değişimin ne kadarını açıkladığını; MAE ortalama mutlak
hatayı; RMSE ise büyük hataları daha fazla cezalandıran hata değerini gösterir.

## Sonuçlar

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Basit regresyon (`total_bill`) | 0.4401 | 0.6522 | 0.8631 |
| Çoklu regresyon | 0.3504 | 0.6975 | 0.9296 |

Bu veri bölünmesinde basit model daha başarılıdır. Cinsiyet, gün, sigara durumu
ve öğün gibi ek değişkenler test setindeki bahşiş değişimini açıklamaya yardım
etmemiştir. Sonuç, daha fazla sütunun her zaman daha iyi model anlamına
gelmediğini gösterir.
