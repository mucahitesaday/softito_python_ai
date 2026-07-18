# Sıcaklık ve Enerji Talebi — Polynomial Regression

Doğrusal olmayan sıcaklık–enerji ilişkisi üzerinde 1, 2, 3, 5 ve 8. derece
polinom modelleri karşılaştırılır. Eğitim ve test hatalarının birlikte
incelenmesiyle underfitting ve overfitting gösterilir.

## Veri seti

360 günlük sıcaklık, nem ve enerji talebi kaydı sabit random seed ile üretilir.
Enerji talebi, konfor sıcaklığı olan 19°C'den uzaklaştıkça ısıtma veya soğutma
ihtiyacı nedeniyle yükselir. Üretilen veri `data/energy_demand.csv` olarak
kaydedilir.

## Yapılan işlemler

- Boyut, eksik değer, tekrar ve tanımlayıcı istatistik kontrolü
- Sıcaklık–enerji dağılımının incelenmesi
- PolynomialFeatures, StandardScaler ve LinearRegression Pipeline'ı
- 1, 2, 3, 5 ve 8. derece için train/test MAE, RMSE ve R²
- Bias–variance ve artık analizi

## Sonuç

| Derece | Test RMSE | Test R² |
|---:|---:|---:|
| 1 | 56.31 | 0.0009 |
| 2 | 9.40 | 0.9722 |
| 3 | 8.55 | 0.9770 |
| 5 | 8.55 | 0.9770 |
| 8 | 8.57 | 0.9769 |

Birinci derece ilişkiyi yakalayamazken üçüncü derece en düşük test hatasını
vermiştir. Daha yüksek dereceler eğitim hatasını azaltsa da test başarısını
artırmamıştır.

```bash
python MachineLearning/Supervised/03_classic_ml/01_polynomial_regression/energy_polynomial.py
```
