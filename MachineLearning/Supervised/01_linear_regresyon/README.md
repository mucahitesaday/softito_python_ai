# 01 — Linear Regression

Doğrusal regresyon, sayısal ve sürekli bir hedefi tahmin eder. Katsayıları
sayesinde her değişkenin tahmin üzerindeki yönü ve yaklaşık etkisi
yorumlanabilir.

Bu bölümde iki farklı gerçek veri seti kullanılır:

| Proje | Amaç | Veri |
|---|---|---|
| Restaurant Tip Prediction | Restoran hesabından bahşiş miktarı tahmini | Seaborn Tips |
| Diabetes Progression | Klinik ölçümlerden hastalık ilerleme skoru tahmini | Scikit-learn Diabetes |

İki projede de:

- Eğitim ve test verisi ayrılır
- Basit ve çoklu regresyon karşılaştırılır
- R², MAE ve RMSE hesaplanır
- Model katsayıları yorumlanır
- Gerçek–tahmin ve artık grafikleri oluşturulur
- Sonuçlar JSON/CSV olarak kaydedilir

## Sonuç özeti

| Proje | Basit model R² | Çoklu model R² | Yorum |
|---|---:|---:|---|
| Bahşiş tahmini | 0.4401 | 0.3504 | Ek kategorik değişkenler test başarısını artırmadı |
| Diyabet ilerleme | 0.2334 | 0.4526 | On özellik birlikte daha açıklayıcı oldu |

Bu iki sonuç önemli bir makine öğrenmesi dersini gösterir: daha fazla özellik
eklemek modeli otomatik olarak iyileştirmez. Yeni değişkenlerin gerçekten bilgi
taşıması ve test verisinde genellenebilmesi gerekir.

## Çalıştırma

```bash
python MachineLearning/Supervised/01_linear_regresyon/restaurant_tip_prediction/tip_regression.py
python MachineLearning/Supervised/01_linear_regresyon/diabetes_progression/diabetes_regression.py
```

Test:

```bash
python -m unittest MachineLearning/Supervised/01_linear_regresyon/test_linear_regresyon.py -v
```
