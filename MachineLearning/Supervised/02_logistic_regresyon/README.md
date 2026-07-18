# 02 — Logistic Regression

Logistic Regression, adına rağmen bir sınıflandırma algoritmasıdır. Doğrusal
bir karar fonksiyonunu sigmoid veya softmax ile olasılığa dönüştürür. Model
katsayılarının incelenebilmesi, tahminlerin açıklanmasını kolaylaştırır.

Bu bölüm iki farklı sınıflandırma türünü kapsar:

| Proje | Sınıflandırma | Veri seti | Hedef |
|---|---|---|---|
| Breast Cancer Diagnosis | İkili | Wisconsin Diagnostic Breast Cancer | İyi/kötü huylu |
| Wine Class Classification | Çok sınıflı | Wine Recognition | Üç şarap sınıfı |

## Uygulanan konular

- Stratified eğitim/test ayrımı
- StandardScaler ve Pipeline kullanımı
- Accuracy, precision, recall ve F1-score
- Confusion matrix
- ROC eğrisi ve ROC-AUC
- Sınıflandırma eşiğinin precision/recall dengesine etkisi
- One-vs-Rest çok sınıflı ROC analizi
- Katsayılarla özellik etkisinin yorumlanması
- 5-fold stratified cross-validation

## Sonuç özeti

| Proje | Accuracy | F1 | ROC-AUC | CV sonucu |
|---|---:|---:|---:|---:|
| Meme tümörü | 0.9720 | 0.9615 | 0.9950 | ROC-AUC: 0.9951 ± 0.0055 |
| Şarap türü | 1.0000 | 1.0000 (macro) | 1.0000 (OvR) | Macro-F1: 0.9829 ± 0.0140 |

Şarap veri setindeki tek test ayrımında bütün örnekler doğru sınıflandırılmıştır.
Cross-validation sonucunun ayrıca verilmesi, başarının yalnızca tek bir veri
bölünmesine bağlı olmadığını kontrol eder.

## Çalıştırma

```bash
python MachineLearning/Supervised/02_logistic_regresyon/breast_cancer_diagnosis/breast_cancer_logistic.py
python MachineLearning/Supervised/02_logistic_regresyon/wine_classification/wine_logistic.py
```

Testler:

```bash
python -m unittest -v MachineLearning/Supervised/02_logistic_regresyon/test_logistic_regresyon.py
```
