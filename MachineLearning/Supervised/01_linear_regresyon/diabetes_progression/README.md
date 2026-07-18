# Diyabet İlerleme Skoru Tahmini

## Amaç

Scikit-learn'in yerleşik Diabetes veri setindeki 442 hastanın on standart klinik
ölçümünü kullanarak, bir yıl sonraki hastalık ilerleme skorunu tahmin etmektir.

İki model karşılaştırılır:

1. Yalnızca BMI kullanan basit doğrusal regresyon
2. On ölçümün tamamını kullanan çoklu doğrusal regresyon

## Uyarı

Bu çalışma yalnızca makine öğrenmesi eğitimi içindir. Tıbbi teşhis, tedavi veya
kişisel sağlık kararı amacıyla kullanılamaz.

Veri kaynağı: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html

## Çalıştırma

```bash
python MachineLearning/Supervised/01_linear_regresyon/diabetes_progression/diabetes_regression.py
```

Projede test seti metriklerine ek olarak 5-fold cross-validation uygulanır.
Model katsayıları, gerçek–tahmin grafiği ve artık dağılımı kaydedilir.

## Sonuçlar

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Basit regresyon (BMI) | 0.2334 | 52.2600 | 63.7325 |
| Çoklu regresyon (10 özellik) | 0.4526 | 42.7941 | 53.8534 |

5-fold cross-validation sonucu `R² = 0.4823 ± 0.0493` olarak ölçülmüştür.
Çoklu model, yalnız BMI kullanan modele göre daha fazla değişimi açıklamıştır;
ancak sonuçlar klinik kullanım veya bireysel tahmin amacı taşımaz.
